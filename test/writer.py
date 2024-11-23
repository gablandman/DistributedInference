import socket

# Sender creates a UDP socket
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Target IP and Port
target_ip = "77.136.67.102"  # Replace with the receiver's public IP
target_port = 5678         # Replace with the receiver's port

# Data to send
message = "Hello, Receiver!"
udp_socket.sendto(message.encode(), (target_ip, target_port))
print(f"Message sent to {target_ip}:{target_port}")

udp_socket.close()


