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


class Edge():
    def __init__(self) -> None:
        self.system_info = get_system_info()
        pass
    def dowload(self, args):
        shard_downloader: ShardDownloader = HFShardDownloader(quick_check=args.download_quick_check,
                                                            max_parallel_downloads=args.max_parallel_downloads) if args.inference_engine != "dummy" else NoopShardDownloader()
        inference_engine_name = args.inference_engine or ("mlx" if self.system_info == "Apple Silicon Mac" else "tinygrad")
        print(f"Inference engine name after selection: {inference_engine_name}")

        inference_engine = get_inference_engine(inference_engine_name, shard_downloader)
        print(f"Using inference engine: {inference_engine.__class__.__name__} with shard downloader: {shard_downloader.__class__.__name__}")
