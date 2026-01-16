#!/usr/bin/env python3
"""
Position Listener for TurtleBot Localizer

Listens to the TCP broadcast from the localizer and provides access to
TurtleBot and reference marker positions.

Similar to the Marvelmind listener but for the rpi_loc localizer.

Usage:
    from rpi_loc.src.position_listener import PositionListener
    
    listener = PositionListener(host="localhost", port=5555)
    listener.start()
    
    # Get latest positions
    positions = listener.get_latest()
    for name, pos in positions.items():
        print(f"{name}: ({pos[0]:.2f}, {pos[1]:.2f})")
    
    listener.stop()
"""

import json
import threading
import time
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.core.networking import TCPClient
from utils.core.setup_logging import setup_logging, get_named_logger

logger = get_named_logger("position_listener", __name__)


class BroadcasterConnectionError(Exception):
    """Raised when connection to the localizer broadcaster is lost."""
    pass


@dataclass
class Position:
    """Position data for a tracked object."""
    x: float
    y: float
    theta: float
    confidence: float
    pixel_x: int
    pixel_y: int
    ts_pub: float
    ts_recv: float


@dataclass
class ReferenceMarker:
    """Reference marker data."""
    marker_id: int
    x: float
    y: float
    pixel_x: int
    pixel_y: int
    detected: bool
    ts_pub: float
    ts_recv: float


class PositionListener:
    """
    Listens to the TurtleBot localizer broadcast.
    
    Maintains history of positions for TurtleBots and reference markers.
    Compatible with the Marvelmind listener interface where applicable.
    
    Args:
        host: Localizer broadcast host
        port: Localizer broadcast port
        history_len: Number of positions to keep in history
        reconnect_delay: Delay between reconnection attempts
        stale_data_timeout: Time after which data is considered stale
        verbose: Enable verbose logging
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        history_len: int = 50,
        reconnect_delay: float = 2.0,
        stale_data_timeout: float = 3.0,
        verbose: bool = False,
    ):
        self.host = host
        self.port = port
        self.reconnect_delay = reconnect_delay
        self.history_len = history_len
        self.stale_data_timeout = stale_data_timeout
        self.verbose = verbose

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # TurtleBot history: key = "M0", "M1", etc.
        # value = deque of Position objects
        self._bot_history: Dict[str, deque] = {}
        
        # Reference marker history: key = "R0", "R1", etc.
        # value = deque of ReferenceMarker objects
        self._ref_history: Dict[str, deque] = {}
        
        # Frame info
        self._last_frame_number: int = 0
        self._homography_valid: bool = False

        self._lock = threading.Lock()

        self._last_heartbeat = 0.0
        self._heartbeat_interval = 10.0  # seconds
        
        # Connection status tracking
        self._connected = False
        self._last_data_time: float = 0.0
        self._connection_error: Optional[Exception] = None
        self._connection_error_lock = threading.Lock()

    def start(self):
        """Start the listener thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Started PositionListener thread. Host={self.host}:{self.port}")

    def stop(self):
        """Stop the listener thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Stopped PositionListener thread.")

    @property
    def connected(self) -> bool:
        """Check if currently connected to broadcaster."""
        return self._connected
    
    @property
    def data_fresh(self) -> bool:
        """Check if data has been received within stale_data_timeout."""
        if self._last_data_time == 0.0:
            return False
        return (time.time() - self._last_data_time) < self.stale_data_timeout
    
    @property
    def homography_valid(self) -> bool:
        """Check if homography is valid (world coordinates are reliable)."""
        with self._lock:
            return self._homography_valid
    
    @property
    def frame_number(self) -> int:
        """Get the last received frame number."""
        with self._lock:
            return self._last_frame_number
    
    def clear_error(self) -> None:
        """
        Clear stored connection error.
        Call this when resuming experiment to allow fresh connection attempt.
        """
        with self._connection_error_lock:
            self._connection_error = None
    
    def check_connection(self) -> None:
        """
        Check connection status and raise BroadcasterConnectionError if disconnected
        or data is stale.
        """
        # Check for stored connection error
        with self._connection_error_lock:
            if self._connection_error is not None:
                err = self._connection_error
                self._connection_error = None  # Clear after raising
                raise BroadcasterConnectionError(f"Broadcaster connection error: {err}")
        
        # Check for stale data
        if self._running and self._last_data_time > 0:
            time_since_data = time.time() - self._last_data_time
            if time_since_data > self.stale_data_timeout:
                raise BroadcasterConnectionError(
                    f"No data received from broadcaster for {time_since_data:.1f}s "
                    f"(timeout: {self.stale_data_timeout}s). Broadcaster may have stopped."
                )

    def get_latest(
        self, 
        check_connection: bool = True
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Get the latest position (x, y, z) for each TurtleBot AND reference marker.
        
        This method returns both mobile objects (TurtleBots) and reference markers
        for compatibility with TB3Manager which needs reference markers for world mapping.
        
        Args:
            check_connection: Whether to check connection status
            
        Returns:
            Dict mapping name to (x, y, z) tuple:
            - TurtleBots: "M0", "M1", ... -> (x, y, theta)
            - Reference markers: "R0", "R1", ... -> (x, y, 0.0)
            
        Raises:
            BroadcasterConnectionError: If data is stale or connection lost.
        """
        if check_connection:
            self.check_connection()
            
        with self._lock:
            result = {}
            
            # Add TurtleBots (M0, M1, ...)
            for name, pos in self._bot_history.items():
                if pos:
                    result[name] = (pos[-1].x, pos[-1].y, pos[-1].theta)
            
            # Add reference markers (R0, R1, ...) - only if detected
            for name, ref in self._ref_history.items():
                if ref and ref[-1].detected:
                    result[name] = (ref[-1].x, ref[-1].y, 0.0)
            
            return result
    
    def get_latest_with_confidence(
        self, 
        check_connection: bool = True
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Get the latest position with confidence for each TurtleBot.
        
        Returns:
            Dict mapping bot name to (x, y, theta, confidence) tuple.
        """
        if check_connection:
            self.check_connection()
            
        with self._lock:
            return {
                name: (pos[-1].x, pos[-1].y, pos[-1].theta, pos[-1].confidence)
                for name, pos in self._bot_history.items()
                if pos
            }
    
    def get_latest_full(
        self, 
        check_connection: bool = True
    ) -> Dict[str, Position]:
        """
        Get the full latest Position object for each TurtleBot.
        
        Returns:
            Dict mapping bot name to Position dataclass.
        """
        if check_connection:
            self.check_connection()
            
        with self._lock:
            return {
                name: pos[-1]
                for name, pos in self._bot_history.items()
                if pos
            }
    
    def get_bot_history(self, name: str) -> List[Position]:
        """
        Get position history for a specific TurtleBot.
        
        Args:
            name: Bot name ("M0", "M1", etc.)
            
        Returns:
            List of Position objects (oldest first).
        """
        with self._lock:
            return list(self._bot_history.get(name, []))
    
    def get_reference_markers(
        self, 
        check_connection: bool = True
    ) -> Dict[str, Tuple[float, float, bool]]:
        """
        Get the latest reference marker positions.
        
        Returns:
            Dict mapping marker name ("R0", "R1", ...) to (x, y, detected) tuple.
        """
        if check_connection:
            self.check_connection()
            
        with self._lock:
            return {
                name: (ref[-1].x, ref[-1].y, ref[-1].detected)
                for name, ref in self._ref_history.items()
                if ref
            }
    
    def get_bot_count(self) -> int:
        """Get the number of currently tracked TurtleBots."""
        with self._lock:
            return len(self._bot_history)
    
    def get_bot_names(self) -> List[str]:
        """Get the names of all tracked TurtleBots."""
        with self._lock:
            return list(self._bot_history.keys())

    def _run(self):
        """Main listener loop using TCPClient."""
        client = TCPClient(
            host=self.host,
            port=self.port,
            timeout=5.0,
            read_timeout=3.0,  # Read timeout to detect stale data
            reconnect=True,
            reconnect_delay=self.reconnect_delay
        )
        
        while self._running:
            try:
                # Attempt to connect
                if not client.connect():
                    self._connected = False
                    with self._connection_error_lock:
                        self._connection_error = ConnectionRefusedError(
                            f"Could not connect to {self.host}:{self.port}"
                        )
                    if self.verbose:
                        logger.warning(f"Connection refused to {self.host}:{self.port}")
                    time.sleep(self.reconnect_delay)
                    continue
                
                self._connected = True
                # Clear any previous connection error on successful connect
                with self._connection_error_lock:
                    self._connection_error = None
                logger.info(f"Connected to localizer broadcaster at {self.host}:{self.port}")

                consecutive_timeouts = 0
                max_consecutive_timeouts = 3  # Reconnect after this many timeouts
                
                while self._running and client.is_connected:
                    line = client.readline()
                    
                    if line is None:
                        # Read timeout or error - connection might still be valid
                        consecutive_timeouts += 1
                        if consecutive_timeouts >= max_consecutive_timeouts:
                            logger.warning(
                                f"No data from broadcaster for {consecutive_timeouts} read cycles, reconnecting...")
                            break
                        # Continue waiting for data
                        continue
                    
                    if line == "":
                        # Connection closed cleanly
                        logger.warning("Connection closed by broadcaster.")
                        err = ConnectionError("Connection closed by broadcaster")
                        with self._connection_error_lock:
                            self._connection_error = err
                        break
                    
                    # Got valid data, reset timeout counter
                    consecutive_timeouts = 0

                    try:
                        payload = json.loads(line)
                        self._handle_payload(payload)
                        self._last_data_time = time.time()
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON received: {e}")
                        continue

                    now = time.monotonic()
                    if now - self._last_heartbeat >= self._heartbeat_interval:
                        if self.verbose:
                            logger.debug("Listener heartbeat: receiving from broadcaster.")
                        self._last_heartbeat = now

            except Exception as e:
                with self._connection_error_lock:
                    self._connection_error = e
                logger.warning(f"Listener error: {e}")
                time.sleep(self.reconnect_delay)

            finally:
                self._connected = False
                client.disconnect()

    def _handle_payload(self, payload: dict):
        """
        Handle incoming payload from broadcaster.
        
        Expected payload format:
        {
            "ts_pub": float,
            "frame_number": int,
            "frame_timestamp": float,
            "homography_valid": bool,
            "reference_markers": {
                "R0": {"marker_id": int, "x": float, "y": float, "pixel": {"x": int, "y": int}, "detected": bool},
                ...
            },
            "turtlebots": {
                "M0": {"local_id": int, "x": float, "y": float, "theta": float, "confidence": float, "pixel": {"x": int, "y": int}},
                ...
            }
        }
        """
        ts_pub = payload.get("ts_pub", time.time())
        ts_recv = time.time()
        
        with self._lock:
            self._last_frame_number = payload.get("frame_number", 0)
            self._homography_valid = payload.get("homography_valid", False)
            
            # Process reference markers
            ref_markers = payload.get("reference_markers", {})
            for name, data in ref_markers.items():
                if name not in self._ref_history:
                    self._ref_history[name] = deque(maxlen=self.history_len)
                
                pixel = data.get("pixel", {})
                self._ref_history[name].append(ReferenceMarker(
                    marker_id=data.get("marker_id", 0),
                    x=data.get("x", 0.0),
                    y=data.get("y", 0.0),
                    pixel_x=pixel.get("x", 0),
                    pixel_y=pixel.get("y", 0),
                    detected=data.get("detected", False),
                    ts_pub=ts_pub,
                    ts_recv=ts_recv
                ))
            
            # Process turtlebots
            turtlebots = payload.get("turtlebots", {})
            
            # Track which bots are in this frame
            current_bots = set(turtlebots.keys())
            
            for name, data in turtlebots.items():
                if name not in self._bot_history:
                    self._bot_history[name] = deque(maxlen=self.history_len)
                    logger.info(f"New TurtleBot detected: {name}")
                
                pixel = data.get("pixel", {})
                self._bot_history[name].append(Position(
                    x=data.get("x", 0.0),
                    y=data.get("y", 0.0),
                    theta=data.get("theta", 0.0),
                    confidence=data.get("confidence", 0.0),
                    pixel_x=pixel.get("x", 0),
                    pixel_y=pixel.get("y", 0),
                    ts_pub=ts_pub,
                    ts_recv=ts_recv
                ))
            
            # Clean up bots that haven't been seen for a while
            stale_bots = []
            for name in self._bot_history:
                if name not in current_bots:
                    # Check if last seen is too old
                    if self._bot_history[name]:
                        last_seen = self._bot_history[name][-1].ts_recv
                        if ts_recv - last_seen > self.stale_data_timeout * 2:
                            stale_bots.append(name)
            
            for name in stale_bots:
                del self._bot_history[name]
                logger.info(f"TurtleBot removed (stale): {name}")


def main():
    """Test the position listener."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the position listener")
    parser.add_argument("--host", default="localhost", help="Broadcaster host")
    parser.add_argument("--port", type=int, default=5555, help="Broadcaster port")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # Setup logging using rpi_loc's setup_logging
    setup_logging(
        experiment_name="pos_listener",
        level="debug" if args.verbose else "info",
        log_to_file=False,
        log_to_console=True
    )
    
    listener = PositionListener(
        host=args.host,
        port=args.port,
        verbose=args.verbose
    )
    
    listener.start()
    print(f"Listening to {args.host}:{args.port}...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(0.5)
            
            if not listener.connected:
                print("Waiting for connection...")
                continue
            
            if not listener.data_fresh:
                print("Waiting for data...")
                continue
            
            # Get positions
            positions = listener.get_latest(check_connection=False)
            refs = listener.get_reference_markers(check_connection=False)
            
            # Clear screen and print
            print("\033[2J\033[H", end="")  # Clear screen
            print(f"Frame: {listener.frame_number} | Homography: {'OK' if listener.homography_valid else 'NO'}")
            print("-" * 40)
            
            print("Reference Markers:")
            for name, (x, y, detected) in sorted(refs.items()):
                status = "✓" if detected else "✗"
                print(f"  {name}: ({x:.2f}, {y:.2f}) [{status}]")
            
            print("\nTurtleBots:")
            if positions:
                for name, (x, y, theta) in sorted(positions.items()):
                    print(f"  {name}: ({x:.2f}, {y:.2f}) θ={theta:.2f}")
            else:
                print("  (none detected)")
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
