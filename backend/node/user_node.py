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

from api import Api

class UserNode(Node):
  def __init__(
    self,
    _id: str, # Api is not defined in this file
    inference_engine: InferenceEngine,
    max_generate_tokens: int = 1024,
    shard_downloader: Optional[HFShardDownloader] = None,
  ):
    self.id = _id
    self.inference_engine = inference_engine
    self.device_capabilities = device_capabilities()
    self.buffered_token_output: Dict[str, Tuple[List[int], bool]] = {}
    self.buffered_logits: Dict[str, List[np.ndarray]] = {}
    self.buffered_inputs: Dict[str, List[np.ndarray]] = {}
    self.max_generate_tokens = max_generate_tokens
    self._on_token = AsyncCallbackSystem[str, Tuple[str, List[int], bool]]()
    self._on_opaque_status = AsyncCallbackSystem[str, Tuple[str, str]]()
    self._on_opaque_status.register("node_status").on_next(self.on_node_status)
    self.shard_downloader = shard_downloader

  async def start(self) -> None:
    await self.server.start()

  async def stop(self) -> None:
    await self.server.stop()

  def on_node_status(self, request_id, opaque_status):
    try:
      status_data = json.loads(opaque_status)
      if status_data.get("type", "") == "supported_inference_engines":
        node_id = status_data.get("node_id")
        engines = status_data.get("engines", [])
        self.api.topology_inference_engines_pool.append(engines) # create topology_inference_engines_pool in api
      download_progress = None
      if status_data.get("type", "") == "download_progress":
        if DEBUG >= 8: print(f"Download progress from {status_data.get('node_id')}: {status_data.get('progress')}")
        self.api.node_download_progress[status_data.get('node_id')] = download_progress   # create node_download_progress in api
    except Exception as e:
      if DEBUG >= 1: print(f"Error updating visualization: {e}")
      if DEBUG >= 1: traceback.print_exc()

  def get_supported_inference_engines(self):
    supported_engine_names = []
    if self.inference_engine.__class__.__name__ == 'MLXDynamicShardInferenceEngine':
      supported_engine_names.append('mlx')
      supported_engine_names.append('tinygrad')
    else:
      supported_engine_names.append('tinygrad')
    return supported_engine_names

  async def broadcast_supported_engines(self, supported_engines_names: List[str]):
    status_message = json.dumps({"type": "supported_inference_engines", "node_id": self.id, "engines": supported_engines_names})
    await self.broadcast_opaque_status("", status_message)

  def get_topology_inference_engines(self) -> None:
    pass
  
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
      self.api.have_finished_request(request_id, self.buffered_token_output[request_id]) # create have_finished_request in api
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
    request_id: str
  ) -> None:
    if DEBUG >= 1: print(f"need to forward prompt: {base_shard} {prompt} {request_id}")
    next_shard = self.api.get_current_shard(base_shard) # create get_current_shard in api
    if DEBUG >= 2: print(f"Computed target from: {base_shard}. next shard: {next_shard}")
    ##### METHOD CAN MAKE ERROR #####
    #await self.process_prompt(next_shard, prompt, request_id)
    if DEBUG >= 1: print(f"Sending prompt to server: {prompt}")
    await self.api.send_prompt(next_shard, prompt, request_id=request_id) # create send_prompt in api

  async def forward_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: str
  ) -> None:
    if DEBUG >= 1: print(f"need to forward prompt: {base_shard} {tensor} {request_id}")
    next_shard = self.api.get_current_shard(base_shard)
    if DEBUG >= 2: print(f"Computed target from: {base_shard}. target shard: {next_shard}")
    ##### METHOD CAN MAKE ERROR #####
    #await self.process_tensor(next_shard, tensor, request_id)
    if DEBUG >= 1: print(f"Sending tensor to server: {tensor}")
    await self.api.send_tensor(next_shard, tensor, request_id=request_id) # create send_tensor in api


  async def select_best_inference_engine(self):
    supported_engines = self.get_supported_inference_engines()
    await self.broadcast_supported_engines(supported_engines)
    if len(self.get_topology_inference_engines()):
      self.inference_engine = get_inference_engine(supported_engines[0], self.shard_downloader)


  async def get_inference_result(self, request_id: str) -> Tuple[Optional[np.ndarray], bool]:
    if request_id not in self.buffered_token_output:
      return None, False
    return np.array(self.buffered_token_output[request_id][0]), self.buffered_token_output[request_id][1]

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
    async def send_result_to_peer(peer):
      try:
        await asyncio.wait_for(peer.send_result(request_id, result, is_finished), timeout=15.0)
      except asyncio.TimeoutError:
        print(f"Timeout broadcasting result to {peer.id()}")
      except Exception as e:
        print(f"Error broadcasting result to {peer.id()}: {e}")
        traceback.print_exc()

    await asyncio.gather(*[send_result_to_peer(peer) for peer in self.peers], return_exceptions=True)

  async def broadcast_opaque_status(self, request_id: str, status: str) -> None:
    if DEBUG >= 8: print(f"Broadcasting opaque status: {request_id=} {status=}")

    async def send_status_to_server(self):
      try:
        await asyncio.wait_for(self.api.send_opaque_status(request_id, status), timeout=15.0) # create send_opaque_status in api
      except asyncio.TimeoutError:
        print(f"Timeout sending opaque status to server")
      except Exception as e:
        print(f"Error sending opaque status to server : {e}")
        traceback.print_exc()

    await asyncio.gather(send_status_to_server(self), return_exceptions=True)
    # in the case of opaque status, we also want to receive our own opaque statuses
    self.on_opaque_status.trigger_all(request_id, status)

  @property
  def current_topology(self) -> None:
    pass
