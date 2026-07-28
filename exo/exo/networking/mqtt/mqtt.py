import asyncio
import json
import time
from typing import List, Dict, Callable, Tuple, Coroutine
from exo.networking.peer_handle import PeerHandle
from exo.topology.device_capabilities import (
    DeviceCapabilities,
    device_capabilities,
    UNKNOWN_DEVICE_CAPABILITIES,
)
from exo.helpers import DEBUG, DEBUG_DISCOVERY
import asyncio_mqtt as mqtt  # Use the asyncio-mqtt library

DEBUG_DISCOVERY = 4  # Adjust this to control debug verbosity


class MQTTDiscovery:
    def __init__(
        self,
        node_id: str,
        mqtt_broker: str,
        mqtt_port: int,
        topic: str,
        known_peers: Dict[str, Tuple[str, int]],
        peer_handle_creator: Callable,
        discovery_timeout: float,
    ):
        self.node_id = node_id
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.topic = topic
        self.known_peers = known_peers  # A dictionary of known peers
        self.peer_handle_creator = peer_handle_creator
        self.discovery_timeout = discovery_timeout
        self.device_capabilities = device_capabilities()
        self.client = mqtt.Client()

    async def start(self):
        self.cleanup_task = asyncio.create_task(self.task_cleanup_peers())
        self.listen_task = asyncio.create_task(self.task_listen_for_peers())
        self.broadcast_task = asyncio.create_task(self.task_broadcast_presence())

    async def stop(self):
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.listen_task:
            self.listen_task.cancel()
        if self.broadcast_task:
            self.broadcast_task.cancel()
        await self.client.disconnect()

    async def task_broadcast_presence(self):
        if DEBUG_DISCOVERY >= 2:
            print("Starting MQTT broadcast presence task...")

        while True:
            message = json.dumps(
                {
                    "type": "discovery",
                    "node_id": self.node_id,
                    "grpc_port": None,
                    "device_capabilities": self.device_capabilities.to_dict(),
                    "priority": 1,
                }
            )
            if DEBUG_DISCOVERY >= 3:
                print(f"Publishing discovery message to {self.topic}: {message}")

            try:
                await self.client.publish(self.topic, message, qos=1)
            except Exception as e:
                if DEBUG_DISCOVERY >= 1:
                    print(f"Error broadcasting presence: {e}")
            await asyncio.sleep(self.discovery_timeout)

    async def task_listen_for_peers(self):
        try:
            async with self.client as client:
                await client.subscribe(self.topic)
                if DEBUG_DISCOVERY >= 2:
                    print(f"Subscribed to MQTT topic: {self.topic}")

                async for message in client.messages:
                    await self.on_listen_message(message.payload)
        except Exception as e:
            if DEBUG_DISCOVERY >= 1:
                print(f"Error listening for peers: {e}")

    async def on_listen_message(self, payload: bytes):
        try:
            decoded_message = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as e:
            if DEBUG_DISCOVERY >= 2:
                print(f"Received invalid JSON data: {e}")
            return

        if DEBUG_DISCOVERY >= 2:
            print(f"Received message: {decoded_message}")

        if (
            decoded_message["type"] == "discovery"
            and decoded_message["node_id"] != self.node_id
        ):
            peer_id = decoded_message["node_id"]
            device_capabilities = DeviceCapabilities(
                **decoded_message["device_capabilities"]
            )

            if peer_id not in self.known_peers:
                new_peer_handle = self.peer_handle_creator(peer_id, device_capabilities)
                if not await new_peer_handle.health_check():
                    if DEBUG_DISCOVERY >= 1:
                        print(f"Peer {peer_id} is not healthy. Skipping.")
                    return

                self.known_peers[peer_id] = (
                    new_peer_handle,
                    time.time(),
                    time.time(),
                    decoded_message.get("priority", 1),
                )
                if DEBUG_DISCOVERY >= 1:
                    print(f"Added new peer: {peer_id}")
            else:
                self.known_peers[peer_id] = (
                    self.known_peers[peer_id][0],
                    self.known_peers[peer_id][1],
                    time.time(),
                    decoded_message.get("priority", 1),
                )

    async def task_cleanup_peers(self):
        while True:
            current_time = time.time()
            peers_to_remove = []

            for peer_id, (_, connected_at, last_seen, _) in self.known_peers.items():
                if current_time - last_seen > self.discovery_timeout:
                    peers_to_remove.append(peer_id)

            for peer_id in peers_to_remove:
                del self.known_peers[peer_id]
                if DEBUG_DISCOVERY >= 1:
                    print(f"Removed inactive peer: {peer_id}")

            await asyncio.sleep(self.discovery_timeout)
