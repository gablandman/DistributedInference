import argparse
import asyncio
import signal
import json
import logging
import platform
import os
import sys
import time
import traceback
import uuid
from exo.exo.networking.manual.manual_discovery import ManualDiscovery
from exo.exo.networking.manual.network_topology_config import NetworkTopology
from exo.exo.orchestration.standard_node import StandardNode
from exo.exo.networking.grpc.grpc_server import GRPCServer
from exo.exo.networking.udp.udp_discovery import UDPDiscovery
from exo.exo.networking.tailscale.tailscale_discovery import TailscaleDiscovery
from exo.exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from exo.exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.exo.api import ChatGPTAPI
from exo.exo.download.shard_download import ShardDownloader, RepoProgressEvent, NoopShardDownloader
from exo.exo.download.hf.hf_shard_download import HFShardDownloader
from exo.exo.helpers import print_yellow_exo, find_available_port, DEBUG, get_system_info, get_or_create_node_id, get_all_ip_addresses, terminal_link, shutdown
from exo.exo.inference.shard import Shard
from exo.exo.inference.inference_engine import get_inference_engine, InferenceEngine
from exo.exo.inference.tokenizers import resolve_tokenizer
from exo.exo.orchestration.node import Node
from exo.exo.models import build_base_shard, get_repo
from exo.exo.viz.topology_viz import TopologyViz
from exo.exo.download.hf.hf_helpers import has_hf_home_read_access, has_hf_home_write_access, get_hf_home, move_models_to_hf

from node.server_node import ServerNode
from node.user_node import UserNode

class API():
    def __init__(self,node_server,nodes_users) -> None:
        self.node_server : ServerNode = node_server
        self.nodes_users : list[UserNode] = nodes_users


    def get_current_shard(self, base_shard: Shard, index: Optional[int] = None) -> Shard:
        if index is None:
            index = self.get_partition_index()
        partitions = self.partitioning_strategy.partition(self.topology)
        shards = map_partitions_to_shards(partitions, base_shard.n_layers, base_shard.model_id)
        return shards[index]
    