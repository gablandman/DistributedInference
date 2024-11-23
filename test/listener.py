import socket

# Receiver creates a UDP socket
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind to a port to listen
host_ip = "0.0.0.0"  # Listen on all interfaces
listen_port = 5678   # Must match sender's target port

udp_socket.bind((host_ip, listen_port))
print(f"Listening on port {listen_port}")

# Receive data
while True:
    data, addr = udp_socket.recvfrom(1024)  # Buffer size
    print(f"Received message: {data.decode()} from {addr}")