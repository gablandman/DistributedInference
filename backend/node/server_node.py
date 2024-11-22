import numpy as np
import json
import asyncio
import uuid
import time
import traceback
from typing import List, Dict, Optional, Tuple, Union, Set
from exo.exo.networking import Discovery, PeerHandle, Server
from exo.exo.inference.inference_engine import InferenceEngine, Shard
from exo.exo.orchestration.node import Node
from exo.exo.topology.topology import Topology
from exo.exo.topology.device_capabilities import device_capabilities
from exo.exo.topology.partitioning_strategy import Partition, PartitioningStrategy, map_partitions_to_shards

from exo import DEBUG
from exo.exo.helpers import AsyncCallbackSystem
from exo.exo.viz.topology_viz import TopologyViz
from exo.exo.download.hf.hf_helpers import RepoProgressEvent
from exo.exo.inference.inference_engine import get_inference_engine, InferenceEngine
from exo.exo.download.hf.hf_shard_download import HFShardDownloader

from node.user_node import UserNode

class ServerNode(Node):
  def __init__(
    self,
    _id: str,
    protocole: Server,
    topology_viz: Optional[TopologyViz] = None,
  ):
    self.id = _id
    self.protocole = protocole
    self.table_nodes = Dict[str, Node] # faire une nouvelle classe pour  strategy
    self.buffered_token_output: Dict[str, Tuple[List[int], bool]] = {}
    self.buffered_logits: Dict[str, List[np.ndarray]] = {}
    self.buffered_inputs: Dict[str, List[np.ndarray]] = {}
    self.topology_viz = topology_viz
    self._on_token = AsyncCallbackSystem[str, Tuple[str, List[int], bool]]()
    self._on_opaque_status = AsyncCallbackSystem[str, Tuple[str, str]]()
    self._on_opaque_status.register("node_status").on_next(self.on_node_status)
    self.node_download_progress: Dict[str, RepoProgressEvent] = {}
    self.topology_inference_engines_pool: List[List[str]] = []

  async def start(self) -> None:
    await self.protocole.start()

  async def stop(self) -> None:
    await self.protocole.stop()

  def on_node_status(self, request_id, opaque_status):
    try:
        status_data = json.loads(opaque_status)
        if status_data.get("type", "") == "node_status":
            if status_data.get("status", "").startswith("start_"):
                self.current_topology.active_node_id = status_data.get("node_id")
            elif status_data.get("status", "").startswith("end_"):
                if status_data.get("node_id") == self.current_topology.active_node_id:
                    self.current_topology.active_node_id = None
            download_progress = None
        if status_data.get("type", "") == "download_progress":
            if DEBUG >= 8: print(f"Download progress from {status_data.get('node_id')}: {status_data.get('progress')}")
            download_progress = RepoProgressEvent.from_dict(status_data.get('progress'))
            self.node_download_progress[status_data.get('node_id')] = download_progress
        if self.topology_viz:
            self.topology_viz.update_visualization(self.current_topology, self.partitioning_strategy.partition(self.current_topology), self.id, self.node_download_progress)
    except Exception as e:
      if DEBUG >= 1: print(f"Error updating visualization: {e}")
      if DEBUG >= 1: traceback.print_exc()

  def get_supported_inference_engines(self):
    pass

  async def broadcast_supported_engines(self, supported_engines_names: List[str]):
    status_message = json.dumps({"type": "supported_inference_engines", "node_id": self.id, "engines": supported_engines_names})
    await self.broadcast_opaque_status("", status_message)

  def get_topology_inference_engines(self) -> List[List[str]]:
    return self.topology_inference_engines_pool
  
  async def process_inference_result(
    self,
    shard,
    result: np.ndarray,
    request_id: Optional[str] = None,
  ):
    if request_id not in self.buffered_token_output:
      self.buffered_token_output[request_id] = ([], False)
    is_finished = len(self.buffered_token_output[request_id][0]) >= self.max_generate_tokens
    if shard.is_last_layer() and not is_finished:
      token = await self.inference_engine.sample(result)
      await self.inference_engine.ensure_shard(shard)
      self.buffered_token_output[request_id][0].append(token.item())
      if DEBUG >= 2: print(f"[{request_id}] result size: {result.size}, is finished: {is_finished}, buffered tokens: {len(self.buffered_token_output[request_id][0])}")
      is_finished = token.item() == self.inference_engine.tokenizer.eos_token_id
      forward = token.reshape(1, -1)
      self.trigger_on_token_callbacks(request_id, self.buffered_token_output[request_id][0], is_finished)
      asyncio.create_task(self.broadcast_result(request_id, self.buffered_token_output[request_id][0], is_finished))
    else:
      forward = result

    if is_finished:
      self.buffered_token_output[request_id] = (self.buffered_token_output[request_id][0], True)
    else:
      asyncio.create_task(self.forward_tensor(shard, forward, request_id, self.get_partition_index(offset = 1)))

    return np.array(self.buffered_token_output[request_id][0])

  async def process_prompt(
    self,
    base_shard: Shard,
    prompt: str,
    request_id: Optional[str] = None,
  ) -> Optional[np.ndarray]:
    shard = self.get_current_shard(base_shard)
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "start_process_prompt",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "prompt": prompt,
          "request_id": request_id,
        }),
      )
    )
    start_time = time.perf_counter_ns()
    resp = await self._process_prompt(base_shard, prompt, request_id)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "end_process_prompt",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "prompt": prompt,
          "request_id": request_id,
          "elapsed_time_ns": elapsed_time_ns,
          "result_size": resp.size if resp is not None else 0,
        }),
      )
    )
    return resp

  async def _process_prompt(self, base_shard: Shard, prompt: str, request_id: Optional[str] = None) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    shard = self.get_current_shard(base_shard)

    if DEBUG >= 2: print(f"[{request_id}] process prompt: {base_shard=} {shard=} {prompt=}")
    if not shard.is_first_layer():
      if DEBUG >= 2: print(f"[{request_id}] forwarding to next shard: {base_shard=} {shard=} {prompt=}")
      resp = await self.forward_prompt(shard, prompt, request_id, 0)
      return None
    else:
      result = await self.inference_engine.infer_prompt(request_id, shard, prompt)
      ret = await self.process_inference_result(shard, result, request_id) 
      return result

  async def process_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
  ) -> Optional[np.ndarray]:
    shard = self.get_current_shard(base_shard)
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "start_process_tensor",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "tensor_size": tensor.size,
          "tensor_shape": tensor.shape,
          "request_id": request_id,
        }),
      )
    )
    start_time = time.perf_counter_ns()
    resp = await self._process_tensor(shard, tensor, request_id)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "end_process_tensor",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "request_id": request_id,
          "elapsed_time_ns": elapsed_time_ns,
          "result_size": resp.size if resp is not None else 0,
        }),
      )
    )
    return resp

  async def _process_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
  ) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    shard = self.get_current_shard(base_shard)

    if DEBUG >= 1: print(f"[{request_id}] process_tensor: {tensor.size=} {tensor.shape=}")
    try:
      result = await self.inference_engine.infer_tensor(request_id, shard, tensor)
      ret = await self.process_inference_result(shard, result, request_id) 
      return ret
    except Exception as e:
      print(f"Error processing tensor for shard {shard}: {e}")
      traceback.print_exc()
      return None

  async def forward_prompt(
    self,
    base_shard: Shard,
    prompt: str,
    request_id: str,
    target_index: int,
  ) -> None:
    if DEBUG >= 1: print(f"target partition index: {target_index}")
    target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
    next_shard = self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"Computed target from: {base_shard} {target_index}, {self.topology}. next shard: {next_shard}")
    if target_id == self.id:
      await self.process_prompt(next_shard, prompt, request_id)
    else:
      target_peer = next((p for p in self.peers if p.id() == target_id), None)
      if not target_peer:
        raise ValueError(f"Peer for {target_index} not found")
      if DEBUG >= 1: print(f"Sending prompt to {target_peer.id()}: {prompt}")
      await target_peer.send_prompt(next_shard, prompt, request_id=request_id)
  
  async def forward_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: str,
    target_index: int,
  ) -> None:
    if DEBUG >= 1: print(f"target partition index: {target_index}")
    target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
    next_shard = self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"Computed target from: {base_shard} {target_index}, {self.topology}. target shard: {next_shard}")
    if target_id == self.id:
      await self.process_tensor(next_shard, tensor, request_id)
    else:
      target_peer = next((p for p in self.peers if p.id() == target_id), None)
      if not target_peer:
        raise ValueError(f"Peer for {target_index} not found")
      if DEBUG >= 1: print(f"Sending tensor to {target_peer.id()}: {tensor}")
      await target_peer.send_tensor(next_shard, tensor, request_id=request_id)

  def get_partition_index(self, offset: int = 0):
    if not self.partitioning_strategy:
      if DEBUG >= 1: print("No partitioning strategy found. Skipping forward.")
      return None
    partitions = self.partitioning_strategy.partition(self.topology)
    current_partition_index = next((i for i, p in enumerate(partitions) if p.node_id == self.id), None)
    if current_partition_index is None:
      raise ValueError(f"No current partition found for node: {self.id}")
    return (current_partition_index + offset) % len(partitions)

  def get_current_shard(self, base_shard: Shard, index: Optional[int] = None) -> Shard:
    if index is None:
      index = self.get_partition_index()
    partitions = self.partitioning_strategy.partition(self.topology)
    shards = map_partitions_to_shards(partitions, base_shard.n_layers, base_shard.model_id)
    return shards[index]

  async def update_peers(self, wait_for_peers: int = 0) -> bool:
    next_peers = await self.discovery.discover_peers(wait_for_peers)
    current_peer_ids = {user.id() for user in self.peers}
    next_peer_ids = {user.id() for user in next_peers}
    peers_added = [user for user in next_peers if user.id() not in current_peer_ids]
    peers_removed = [user for user in self.peers if user.id() not in next_peer_ids]
    peers_updated = [user for user in next_peers if user.id() in current_peer_ids and any(p.addr() != user.addr() for p in self.peers if p.id() == user.id())]
    peers_unchanged = [user for user in next_peers if user.id() in current_peer_ids and all(p.addr() == user.addr() for p in self.peers if p.id() == user.id())]
    peers_to_disconnect = [user for user in peers_removed if await user.is_connected()]
    peers_to_connect = [user for user in peers_added + peers_updated + peers_unchanged if not await user.is_connected()]

    def _pretty(peers: List[PeerHandle]) -> List[str]:
      return [f"{user.id()}@{user.addr()}" for user in peers]

    if DEBUG >= 2:
      print(f"update_peers: added={peers_added} removed={peers_removed} updated={peers_updated} unchanged={peers_unchanged} to_disconnect={peers_to_disconnect} to_connect={peers_to_connect}")

    async def disconnect_with_timeout(user, timeout=5):
      try:
        await asyncio.wait_for(user.disconnect(), timeout)
        return True
      except Exception as e:
        print(f"Error disconnecting user {user.id()}@{user.addr()}: {e}")
        traceback.print_exc()
        return False

    async def connect_with_timeout(user, timeout=5):
      try:
        await asyncio.wait_for(user.connect(), timeout)
        return True
      except Exception as e:
        print(f"Error connecting user {user.id()}@{user.addr()}: {e}")
        traceback.print_exc()
        return False

    disconnect_results = await asyncio.gather(*(disconnect_with_timeout(user) for user in peers_to_disconnect), return_exceptions=True)
    connect_results = await asyncio.gather(*(connect_with_timeout(user) for user in peers_to_connect), return_exceptions=True)

    successful_disconnects = [user for user, result in zip(peers_to_disconnect, disconnect_results) if result is True]
    failed_disconnects = [user for user, result in zip(peers_to_disconnect, disconnect_results) if result is False]
    successful_connects = [user for user, result in zip(peers_to_connect, connect_results) if result is True]
    failed_connects = [user for user, result in zip(peers_to_connect, connect_results) if result is False]
    if DEBUG >= 1:
      if successful_disconnects: print(f"Successfully disconnected peers: {_pretty(successful_disconnects)}")
      if failed_disconnects: print(f"Failed to disconnect peers: {_pretty(failed_disconnects)}")
      if successful_connects: print(f"Successfully connected peers: {_pretty(successful_connects)}")
      if failed_connects: print(f"Failed to connect peers: {_pretty(failed_connects)}")

    self.peers = next_peers
    return len(peers_added) > 0 or len(peers_removed) > 0 or len(peers_updated) > 0

  async def select_best_inference_engine(self):
    supported_engines = self.get_supported_inference_engines()
    await self.broadcast_supported_engines(supported_engines)
    if len(self.get_topology_inference_engines()):
      self.inference_engine = get_inference_engine(supported_engines[0], self.shard_downloader)

  async def get_inference_result(self, request_id: str) -> Tuple[Optional[np.ndarray], bool]:
    if request_id not in self.buffered_token_output:
      return None, False
    return np.array(self.buffered_token_output[request_id][0]), self.buffered_token_output[request_id][1]

  async def collect_topology(self, visited: set[str] = set(), max_depth: int = 4) -> Topology:
    next_topology = Topology()
    next_topology.update_node(self.id, self.device_capabilities)

    if DEBUG >= 2: print(f"Collecting topology {max_depth=} {visited=}")

    prev_visited = visited.copy()
    visited.add(self.id)
    visited.update(p.id() for p in self.peers)

    for user in self.peers:
      next_topology.update_node(user.id(), user.device_capabilities())
      next_topology.add_edge(self.id, user.id())

      if user.id() in prev_visited:
        continue

      if max_depth <= 0:
        if DEBUG >= 2: print("Max depth reached. Skipping...")
        continue

      try:
        other_topology = await asyncio.wait_for(user.collect_topology(visited, max_depth=max_depth - 1), timeout=5.0)
        if DEBUG >= 2: print(f"Collected topology from: {user.id()}: {other_topology}")
        self.topology.merge(other_topology)
      except Exception as e:
        print(f"Error collecting topology from {user.id()}: {e}")
        traceback.print_exc()

    next_topology.active_node_id = self.topology.active_node_id  # this is not so clean.
    self.topology = next_topology
    if self.topology_viz:
      self.topology_viz.update_visualization(self.current_topology, self.partitioning_strategy.partition(self.current_topology), self.id)
    return next_topology

  @property
  def on_token(self) -> AsyncCallbackSystem[str, Tuple[str, List[int], bool]]:
    return self._on_token

  @property
  def on_opaque_status(self) -> AsyncCallbackSystem[str, Tuple[str, str]]:
    return self._on_opaque_status

  def trigger_on_token_callbacks(self, request_id: str, tokens: List[int], is_finished: bool) -> None:
    if DEBUG >= 2: print(f"Triggering all on_token callbacks with {request_id=} num_tokens={len(tokens)} {is_finished=}")
    self.on_token.trigger_all(request_id, tokens, is_finished)
  
  async def broadcast_result(self, request_id: str, result: List[int], is_finished: bool) -> None:
    async def send_result_to_peer(user):
      try:
        await asyncio.wait_for(user.send_result(request_id, result, is_finished), timeout=15.0)
      except asyncio.TimeoutError:
        print(f"Timeout broadcasting result to {user.id()}")
      except Exception as e:
        print(f"Error broadcasting result to {user.id()}: {e}")
        traceback.print_exc()

    await asyncio.gather(*[send_result_to_peer(user) for user in self.peers], return_exceptions=True)

  async def broadcast_opaque_status(self, request_id: str, user : UserNode,  status: str) -> None:
    if DEBUG >= 8: print(f"Broadcasting opaque status: {request_id=} {status=}")

    async def send_status_to_user(user):
      try:
        await asyncio.wait_for(user.send_opaque_status(request_id, status), timeout=15.0)
      except asyncio.TimeoutError:
        print(f"Timeout sending opaque status to {user.id()}")
      except Exception as e:
        print(f"Error sending opaque status to {user.id()}: {e}")
        traceback.print_exc()

    await asyncio.gather(send_status_to_user(user), return_exceptions=True)
    # in the case of opaque status, we also want to receive our own opaque statuses
    self.on_opaque_status.trigger_all(request_id, status)

  @property
  def current_topology(self) -> Topology:
    return self.topology
