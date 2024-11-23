import asyncio
from exo.networking.udp.udp_discovery import UDPDiscovery
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from exo.networking.grpc.grpc_server import GRPCServer
from exo.orchestration.node import Node
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
from exo.networking.manual.manual_discovery import ManualDiscovery
from exo.networking.manual.network_topology_config import NetworkTopology
from exo.orchestration.standard_node import StandardNode
from exo.networking.grpc.grpc_server import GRPCServer
from exo.networking.udp.udp_discovery import UDPDiscovery
from exo.networking.tailscale.tailscale_discovery import TailscaleDiscovery
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.api import ChatGPTAPI
from exo.download.shard_download import ShardDownloader, RepoProgressEvent, NoopShardDownloader
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.helpers import print_yellow_exo, find_available_port, DEBUG, get_system_info, get_or_create_node_id, get_all_ip_addresses, terminal_link, shutdown
from exo.inference.shard import Shard
from exo.inference.inference_engine import get_inference_engine, InferenceEngine
from exo.inference.tokenizers import resolve_tokenizer
from exo.orchestration.node import Node
from exo.models import build_base_shard, get_repo
from exo.viz.topology_viz import TopologyViz
from exo.download.hf.hf_helpers import has_hf_home_read_access, has_hf_home_write_access, get_hf_home, move_models_to_hf


def main(args):


    system_info = get_system_info()
    print(f"Detected system: {system_info}")

    shard_downloader: ShardDownloader = HFShardDownloader(quick_check=args.download_quick_check,
                                                        max_parallel_downloads=args.max_parallel_downloads) if args.inference_engine != "dummy" else NoopShardDownloader()
    inference_engine_name = args.inference_engine or ("mlx" if system_info == "Apple Silicon Mac" else "tinygrad")
    print(f"Inference engine name after selection: {inference_engine_name}")

    inference_engine = get_inference_engine(inference_engine_name, shard_downloader)
    print(f"Using inference engine: {inference_engine.__class__.__name__} with shard downloader: {shard_downloader.__class__.__name__}")

    if args.node_port is None:
        args.node_port = find_available_port(args.node_host)

    args.node_id = args.node_id or get_or_create_node_id()
    print(f"get_all_ip_addresses: {get_all_ip_addresses()}")

    discovery = UDPDiscovery(
        args.node_id,
        args.node_port,
        args.listen_port,
        args.broadcast_port,
        lambda peer_id, address, device_capabilities: GRPCPeerHandle(peer_id, address, device_capabilities),
        discovery_timeout=args.discovery_timeout)

    node = StandardNode(
    args.node_id,
    None,
    inference_engine,
    discovery,
    partitioning_strategy=RingMemoryWeightedPartitioningStrategy(),
    max_generate_tokens=args.max_generate_tokens,
    topology_viz=None,
    shard_downloader=shard_downloader
    )
    server = GRPCServer(node, args.node_host, args.node_port)
    node.server = server
    asyncio.run(server.start())
    discovery1 = UDPDiscovery("discovery1", 50053, 5678, 5679, lambda peer_id, address, device_capabilities: GRPCPeerHandle(peer_id, address, device_capabilities))
    asyncio.run(discovery1.start())
    asyncio.run(discovery1.discover_peers())
    asyncio.run(discovery1.stop())
    asyncio.run(server.stop())


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="Initialize GRPC Discovery")
    parser.add_argument("command", nargs="?", choices=["run"], help="Command to run")
    parser.add_argument("model_name", nargs="?", help="Model name to run")
    parser.add_argument("--default-model", type=str, default=None, help="Default model")
    parser.add_argument("--node-id", type=str, default=None, help="Node ID")
    parser.add_argument("--node-host", type=str, default="0.0.0.0", help="Node host")
    parser.add_argument("--node-port", type=int, default=50053, help="Node port")
    parser.add_argument("--models-seed-dir", type=str, default=None, help="Model seed directory")
    parser.add_argument("--listen-port", type=int, default=5678, help="Listening port for discovery")
    parser.add_argument("--download-quick-check", action="store_true", help="Quick check local path for model shards download")
    parser.add_argument("--max-parallel-downloads", type=int, default=4, help="Max parallel downloads for model shards download")
    parser.add_argument("--prometheus-client-port", type=int, default=None, help="Prometheus client port")
    parser.add_argument("--broadcast-port", type=int, default=5678, help="Broadcast port for discovery")
    parser.add_argument("--discovery-module", type=str, choices=["udp", "tailscale", "manual"], default="udp", help="Discovery module to use")
    parser.add_argument("--discovery-timeout", type=int, default=30, help="Discovery timeout in seconds")
    parser.add_argument("--discovery-config-path", type=str, default=None, help="Path to discovery config json file")
    parser.add_argument("--wait-for-peers", type=int, default=0, help="Number of peers to wait to connect to before starting")
    parser.add_argument("--chatgpt-api-port", type=int, default=52415, help="ChatGPT API port")
    parser.add_argument("--chatgpt-api-response-timeout", type=int, default=90, help="ChatGPT API response timeout in seconds")
    parser.add_argument("--max-generate-tokens", type=int, default=10000, help="Max tokens to generate in each request")
    parser.add_argument("--inference-engine", type=str, default=None, help="Inference engine to use (mlx, tinygrad, or dummy)")
    parser.add_argument("--disable-tui", action=argparse.BooleanOptionalAction, help="Disable TUI")
    parser.add_argument("--run-model", type=str, help="Specify a model to run directly")
    parser.add_argument("--prompt", type=str, help="Prompt for the model when using --run-model", default="Who are you?")
    parser.add_argument("--tailscale-api-key", type=str, default=None, help="Tailscale API key")
    parser.add_argument("--tailnet-name", type=str, default=None, help="Tailnet name")
    args = parser.parse_args()
    print(f"Selected inference engine: {args.inference_engine}")
    main(args)