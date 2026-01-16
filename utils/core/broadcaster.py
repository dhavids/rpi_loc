"""
Position Broadcaster

Broadcasts TurtleBot position data over TCP to connected clients.
Uses TCPServer from networking module for connection management.
"""

import json
import threading
import time
import logging
from typing import Dict, Optional, Any

from .networking import TCPServer

logger = logging.getLogger(__name__)


class PositionBroadcaster:
    """
    TCP broadcaster for TurtleBot position data.
    
    Clients can connect to receive JSON-formatted position updates
    at a configurable rate. Built on top of TCPServer for connection handling.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5555,
        rate_hz: float = 20.0
    ):
        """
        Initialize the broadcaster.
        
        Args:
            host: Bind address (0.0.0.0 for all interfaces)
            port: TCP port to listen on
            rate_hz: Maximum broadcast rate in Hz
        """
        self.period = 1.0 / rate_hz
        
        self._server = TCPServer(
            host=host,
            port=port,
            max_clients=10,
            on_connect=self._on_client_connect,
            on_disconnect=self._on_client_disconnect
        )
        
        self._latest_payload: Optional[Dict] = None
        self._broadcast_thread: Optional[threading.Thread] = None
        self._running = False
        
        self._last_log_time = 0.0
        
        # Statistics
        self._messages_sent = 0
        self._bytes_sent = 0
    
    def _on_client_connect(self, sock, addr):
        """Called when a client connects."""
        logger.info(f"Broadcast client connected: {addr[0]}:{addr[1]}")
    
    def _on_client_disconnect(self, sock, addr):
        """Called when a client disconnects."""
        logger.warning(f"Broadcast client disconnected: {addr[0]}:{addr[1]}")
    
    def start(self) -> bool:
        """
        Start the broadcaster.
        
        Returns:
            True if started successfully
        """
        if not self._server.start():
            return False
        
        self._running = True
        self._broadcast_thread = threading.Thread(
            target=self._broadcast_loop, 
            daemon=True
        )
        self._broadcast_thread.start()
        
        logger.info(f"Broadcaster ready at {1.0/self.period:.1f} Hz")
        return True
    
    def stop(self):
        """Stop the broadcaster and close all connections."""
        self._running = False
        self._server.stop()
        
        logger.info(
            f"Broadcaster stopped. Sent {self._messages_sent} messages, "
            f"{self._bytes_sent} bytes"
        )
    
    def update(self, payload: Dict[str, Any]):
        """
        Update the payload to broadcast.
        
        Args:
            payload: Dictionary to broadcast as JSON
        """
        self._latest_payload = payload
    
    def pause(self):
        """Pause broadcasting (keeps connections but stops sending)."""
        self._latest_payload = None
    
    def _broadcast_loop(self):
        """Broadcast payload to all connected clients at configured rate."""
        last_broadcast = 0.0
        
        while self._running:
            now = time.monotonic()
            
            # Rate limiting
            if now - last_broadcast < self.period:
                time.sleep(0.001)
                continue
            
            if self._latest_payload is not None and self._server.client_count > 0:
                msg = json.dumps(self._latest_payload).encode() + b"\n"
                sent_count = self._server.broadcast(msg)
                
                if sent_count > 0:
                    self._messages_sent += 1
                    self._bytes_sent += len(msg) * sent_count
                
                last_broadcast = now
                
                # Throttled logging (once per second)
                if now - self._last_log_time >= 1.0:
                    tb_count = len(self._latest_payload.get("turtlebots", []))
                    logger.debug(
                        f"Broadcasted {tb_count} TurtleBots to {sent_count} clients"
                    )
                    self._last_log_time = now
            else:
                last_broadcast = now
            
            time.sleep(self.period)
    
    @property
    def client_count(self) -> int:
        """Get number of connected clients."""
        return self._server.client_count
    
    @property
    def is_running(self) -> bool:
        """Check if broadcaster is running."""
        return self._running and self._server.is_running
    
    @property
    def port(self) -> int:
        """Get the broadcast port."""
        return self._server.port
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get broadcaster statistics."""
        return {
            "messages_sent": self._messages_sent,
            "bytes_sent": self._bytes_sent,
            "client_count": self.client_count,
            "running": self.is_running
        }
