"""
Common utility functions for RPi localization and image streaming.
"""

import socket
import time
import json
import struct
from datetime import datetime
from typing import Dict, Any, Optional, Tuple


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def get_timestamp_ms() -> int:
    """Get current timestamp in milliseconds since epoch."""
    return int(time.time() * 1000)


def get_hostname() -> str:
    """Get the hostname of the device."""
    return socket.gethostname()


def get_ip_address(interface: str = None) -> str:
    """
    Get the IP address of the device.
    
    Args:
        interface: Optional network interface name (e.g., 'tailscale0', 'eth0')
    
    Returns:
        IP address as string
    """
    try:
        if interface:
            import netifaces
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                return addrs[netifaces.AF_INET][0]['addr']
        
        # Fallback: get default IP by connecting to external address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_tailscale_ip() -> str:
    """Get the Tailscale IP address of the device."""
    try:
        import netifaces
        if 'tailscale0' in netifaces.interfaces():
            return get_ip_address('tailscale0')
    except ImportError:
        pass
    
    # Fallback: try to parse from tailscale CLI
    try:
        import subprocess
        result = subprocess.run(['tailscale', 'ip', '-4'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    return get_ip_address()


def create_metadata(
    device_id: str = None,
    additional_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create metadata dictionary for image frames.
    
    Args:
        device_id: Optional device identifier
        additional_data: Optional additional metadata to include
    
    Returns:
        Dictionary containing metadata
    """
    metadata = {
        "timestamp": get_timestamp(),
        "timestamp_ms": get_timestamp_ms(),
        "hostname": get_hostname(),
        "ip_address": get_tailscale_ip(),
        "device_id": device_id or get_hostname(),
    }
    
    if additional_data:
        metadata.update(additional_data)
    
    return metadata


def pack_frame_data(image_data: bytes, metadata: Dict[str, Any]) -> bytes:
    """
    Pack image data and metadata into a single message for transmission.
    
    Format:
        - 4 bytes: metadata length (uint32, big-endian)
        - N bytes: metadata JSON
        - 4 bytes: image data length (uint32, big-endian)
        - M bytes: image data
    
    Args:
        image_data: Raw image bytes (e.g., JPEG encoded)
        metadata: Metadata dictionary
    
    Returns:
        Packed bytes ready for transmission
    """
    metadata_bytes = json.dumps(metadata).encode('utf-8')
    
    packed = struct.pack('>I', len(metadata_bytes))
    packed += metadata_bytes
    packed += struct.pack('>I', len(image_data))
    packed += image_data
    
    return packed


def unpack_frame_data(data: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """
    Unpack received frame data into image and metadata.
    
    Args:
        data: Packed frame data
    
    Returns:
        Tuple of (image_data, metadata)
    """
    offset = 0
    
    # Read metadata
    metadata_len = struct.unpack('>I', data[offset:offset+4])[0]
    offset += 4
    metadata = json.loads(data[offset:offset+metadata_len].decode('utf-8'))
    offset += metadata_len
    
    # Read image data
    image_len = struct.unpack('>I', data[offset:offset+4])[0]
    offset += 4
    image_data = data[offset:offset+image_len]
    
    return image_data, metadata


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """
    Receive exactly n bytes from a socket.
    
    Args:
        sock: Socket to receive from
        n: Number of bytes to receive
    
    Returns:
        Received bytes
    
    Raises:
        ConnectionError: If connection is closed before all bytes received
    """
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed before all data received")
        data += chunk
    return data


def recv_frame(sock: socket.socket) -> Tuple[bytes, Dict[str, Any]]:
    """
    Receive a complete frame (image + metadata) from a socket.
    
    Args:
        sock: Socket to receive from
    
    Returns:
        Tuple of (image_data, metadata)
    """
    # Read metadata length and metadata
    metadata_len = struct.unpack('>I', recv_exact(sock, 4))[0]
    metadata = json.loads(recv_exact(sock, metadata_len).decode('utf-8'))
    
    # Read image length and image
    image_len = struct.unpack('>I', recv_exact(sock, 4))[0]
    image_data = recv_exact(sock, image_len)
    
    return image_data, metadata


def send_frame(sock: socket.socket, image_data: bytes, metadata: Dict[str, Any]) -> None:
    """
    Send a complete frame (image + metadata) over a socket.
    
    Args:
        sock: Socket to send over
        image_data: Raw image bytes
        metadata: Metadata dictionary
    """
    packed = pack_frame_data(image_data, metadata)
    sock.sendall(packed)


class StreamConfig:
    """Configuration class for image streaming."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        quality: int = 85,
        format: str = "jpeg"
    ):
        self.host = host
        self.port = port
        self.resolution = resolution
        self.fps = fps
        self.quality = quality
        self.format = format
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "resolution": self.resolution,
            "fps": self.fps,
            "quality": self.quality,
            "format": self.format
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamConfig":
        return cls(**data)
