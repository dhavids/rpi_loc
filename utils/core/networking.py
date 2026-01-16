"""
Core networking utilities for RPi localization.
"""

import socket
import threading
import logging
from typing import Callable, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a network connection."""
    address: str
    port: int
    connected_at: float
    bytes_sent: int = 0
    bytes_received: int = 0


class TCPServer:
    """
    Reusable TCP server with connection handling.
    
    Provides a base class for building TCP-based streaming servers.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        max_clients: int = 10,
        on_connect: Optional[Callable[[socket.socket, tuple], None]] = None,
        on_disconnect: Optional[Callable[[socket.socket, tuple], None]] = None
    ):
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        
        self._server_socket: Optional[socket.socket] = None
        self._clients: dict = {}  # socket -> ConnectionInfo
        self._lock = threading.Lock()
        self._running = False
    
    def start(self) -> bool:
        """Start the TCP server."""
        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind((self.host, self.port))
            self._server_socket.listen(self.max_clients)
            self._server_socket.settimeout(1.0)
            
            self._running = True
            
            accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
            accept_thread.start()
            
            logger.info(f"TCP server started on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start TCP server: {e}")
            return False
    
    def stop(self):
        """Stop the TCP server."""
        self._running = False
        
        with self._lock:
            for client_socket in list(self._clients.keys()):
                self._close_client(client_socket)
            self._clients.clear()
        
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None
        
        logger.info("TCP server stopped")
    
    def _accept_loop(self):
        """Accept incoming connections."""
        import time
        
        while self._running:
            try:
                client_socket, addr = self._server_socket.accept()
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                with self._lock:
                    self._clients[client_socket] = ConnectionInfo(
                        address=addr[0],
                        port=addr[1],
                        connected_at=time.time()
                    )
                
                logger.info(f"Client connected: {addr}")
                
                if self.on_connect:
                    self.on_connect(client_socket, addr)
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Accept error: {e}")
    
    def _close_client(self, client_socket: socket.socket):
        """Close a client connection."""
        try:
            info = self._clients.get(client_socket)
            if info and self.on_disconnect:
                self.on_disconnect(client_socket, (info.address, info.port))
            client_socket.close()
        except Exception:
            pass
    
    def broadcast(self, data: bytes) -> int:
        """
        Broadcast data to all connected clients.
        
        Returns:
            Number of clients that received the data
        """
        sent_count = 0
        disconnected = []
        
        with self._lock:
            for client_socket, info in self._clients.items():
                try:
                    client_socket.sendall(data)
                    info.bytes_sent += len(data)
                    sent_count += 1
                except Exception:
                    disconnected.append(client_socket)
            
            for client_socket in disconnected:
                self._close_client(client_socket)
                del self._clients[client_socket]
        
        return sent_count
    
    @property
    def client_count(self) -> int:
        """Get the number of connected clients."""
        with self._lock:
            return len(self._clients)
    
    @property
    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._running


class TCPClient:
    """
    Reusable TCP client for connecting to servers.
    
    Supports both raw byte communication and line-based JSON streaming.
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        timeout: float = 10.0,
        read_timeout: float = 5.0,
        reconnect: bool = True,
        reconnect_delay: float = 5.0
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.read_timeout = read_timeout  # Timeout for read operations
        self.reconnect = reconnect
        self.reconnect_delay = reconnect_delay
        
        self._socket: Optional[socket.socket] = None
        self._file: Optional[Any] = None  # File-like interface for readline
        self._connected = False
        self._lock = threading.Lock()
    
    def connect(self) -> bool:
        """Connect to the server."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((self.host, self.port))
            # Keep a read timeout to prevent blocking indefinitely
            self._socket.settimeout(self.read_timeout)
            self._file = self._socket.makefile("r")
            self._connected = True
            logger.info(f"Connected to {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the server."""
        self._connected = False
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
    
    def send(self, data: bytes) -> bool:
        """Send data to the server."""
        if not self._connected:
            return False
        
        try:
            with self._lock:
                self._socket.sendall(data)
            return True
        except Exception as e:
            logger.error(f"Send failed: {e}")
            self._connected = False
            return False
    
    def receive(self, size: int) -> Optional[bytes]:
        """Receive exactly `size` bytes from the server."""
        if not self._connected:
            return None
        
        try:
            data = b''
            while len(data) < size:
                chunk = self._socket.recv(size - len(data))
                if not chunk:
                    raise ConnectionError("Connection closed")
                data += chunk
            return data
        except Exception as e:
            logger.error(f"Receive failed: {e}")
            self._connected = False
            return None
    
    def readline(self) -> Optional[str]:
        """
        Read a line from the server (for line-based protocols like JSON streams).
        
        Returns:
            The line as a string (including newline), or None if disconnected or timeout.
            Empty string means connection was closed by server.
        """
        if not self._connected or not self._file:
            return None
        
        try:
            line = self._file.readline()
            if not line:
                # Connection closed
                self._connected = False
                return ""
            return line
        except socket.timeout:
            # Read timeout - connection might still be valid, return None to trigger retry
            logger.warning(f"Read timeout on {self.host}:{self.port}")
            return None
        except Exception as e:
            logger.error(f"Readline failed: {e}")
            self._connected = False
            return None
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to the server."""
        return self._connected


def get_available_port(start: int = 5000, end: int = 6000) -> Optional[int]:
    """
    Find an available port in the given range.
    
    Args:
        start: Start of port range
        end: End of port range
    
    Returns:
        Available port number or None if no port available
    """
    for port in range(start, end):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            continue
    return None


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """
    Check if a port is open on a host.
    
    Args:
        host: Host address
        port: Port number
        timeout: Connection timeout
    
    Returns:
        True if port is open
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False
