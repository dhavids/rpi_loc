#!/usr/bin/env python3
"""
Image Listener

Receives and displays images from an ImageStreamer running on a Raspberry Pi.
This script is designed to run on the receiving machine (not on the RPi).

Usage:
    python image_listener.py --host <rpi_tailscale_ip> --port 5000
    python image_listener.py --host 100.x.x.x --port 5000 --save ./frames

Requirements:
    - opencv-python
    - numpy
"""

import argparse
import logging
import signal
import sys
import socket
import os
from typing import Optional, Callable, Dict, Any

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from utils.common import recv_frame
from utils.core.setup_logging import setup_logging, get_named_logger

logger = get_named_logger("listener", __name__)


class ImageListener:
    """
    Receives and processes images from an ImageStreamer.
    
    This class connects to a remote ImageStreamer server and receives
    frames along with their metadata. It can display frames, save them
    to disk, and/or pass them to a custom callback for processing.
    
    Attributes:
        host: Remote host address (Tailscale IP of the streamer)
        port: Remote port number
        display: Whether to display received frames in a window
        save_dir: Optional directory to save received frames
        on_frame: Optional callback function called for each received frame
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        display: bool = True,
        save_dir: str = None,
        on_frame: Optional[Callable[[np.ndarray, Dict[str, Any]], None]] = None,
        window_name: str = None,
        max_display_width: int = 1280,
        max_display_height: int = 720,
        show_overlay: bool = False,
        max_lag_ms: float = 200.0
    ):
        """
        Initialize the ImageListener.
        
        Args:
            host: Remote host address (e.g., Tailscale IP)
            port: Remote port number
            display: Whether to display frames in an OpenCV window
            save_dir: Directory to save frames (None to disable saving)
            on_frame: Callback function(frame, metadata) for custom processing
            window_name: Custom window name for display
            max_display_width: Maximum display window width (scales down if larger)
            max_display_height: Maximum display window height (scales down if larger)
            show_overlay: Whether to show text overlay on display
            max_lag_ms: Maximum acceptable lag in ms - frames older than this are dropped
        """
        self.host = host
        self.port = port
        self.display = display
        self.save_dir = save_dir
        self.on_frame = on_frame
        self.window_name = window_name
        self.max_display_width = max_display_width
        self.max_display_height = max_display_height
        self.show_overlay = show_overlay
        self.max_lag_ms = max_lag_ms
        
        self._socket: Optional[socket.socket] = None
        self._running = False
        self._frame_count = 0
        self._frames_dropped = 0
        self._connected = False
        
        # Statistics
        self._bytes_received = 0
        self._last_frame_time = 0
        self._fps = 0.0
        self._last_server_frame = 0  # Last frame number from server
        self._lag_ms = 0.0  # Estimated lag in milliseconds
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Saving frames to: {save_dir}")
    
    def connect(self) -> bool:
        """
        Connect to the remote ImageStreamer.
        
        Returns:
            True if connection successful, False otherwise
        """
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(10.0)
        
        # Reduce receive buffer to minimize latency (default is often 128KB+)
        # Smaller buffer = less queued data = lower latency
        try:
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)  # 64KB
        except Exception:
            pass
        
        try:
            logger.info(f"Connecting to {self.host}:{self.port}...")
            self._socket.connect((self.host, self.port))
            self._socket.settimeout(None)  # Remove timeout for streaming
            # Disable Nagle's algorithm for lower latency
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._connected = True
            logger.info("Connected successfully!")
            return True
        except socket.timeout:
            logger.error("Connection timed out")
            return False
        except ConnectionRefusedError:
            logger.error("Connection refused - is the streamer running?")
            return False
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the remote streamer."""
        self._connected = False
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        logger.info("Disconnected from streamer")
    
    def start(self, auto_reconnect: bool = False, reconnect_delay: float = 5.0) -> bool:
        """
        Start listening for frames.
        
        This method blocks until stopped or connection is lost.
        
        Args:
            auto_reconnect: Whether to automatically reconnect on connection loss
            reconnect_delay: Seconds to wait before reconnecting
        
        Returns:
            True if started successfully
        """
        import time
        
        if not self._connected:
            if not self.connect():
                return False
        
        self._running = True
        
        try:
            while self._running:
                try:
                    self._receive_loop()
                except ConnectionError as e:
                    logger.warning(f"Connection lost: {e}")
                    self._connected = False
                    
                    if auto_reconnect and self._running:
                        logger.info(f"Reconnecting in {reconnect_delay} seconds...")
                        time.sleep(reconnect_delay)
                        if not self.connect():
                            continue
                    else:
                        break
                        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
        
        return True
    
    def _receive_loop(self):
        """Main loop for receiving frames."""
        import time
        
        while self._running and self._connected:
            # Receive frame
            image_data, metadata = recv_frame(self._socket)
            
            self._bytes_received += len(image_data)
            
            # Calculate FPS and lag
            current_time = time.time()
            if self._last_frame_time > 0:
                dt = current_time - self._last_frame_time
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if dt > 0 else self._fps
            self._last_frame_time = current_time
            
            # Track server frame number and estimate lag from timestamp
            server_frame = metadata.get('frame_number', 0)
            self._last_server_frame = server_frame
            server_timestamp_ms = metadata.get('timestamp_ms', 0)
            if server_timestamp_ms > 0:
                self._lag_ms = (current_time * 1000) - server_timestamp_ms
            
            # Drop frames that are too old (lag > threshold)
            if self.max_lag_ms > 0 and self._lag_ms > self.max_lag_ms:
                self._frames_dropped += 1
                # Log occasionally when dropping
                if self._frames_dropped % 30 == 1:
                    logger.warning(f"Dropping frames (lag={self._lag_ms:.0f}ms > {self.max_lag_ms:.0f}ms) - "
                                   f"dropped {self._frames_dropped} total")
                continue
            
            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.warning("Failed to decode frame")
                continue
            
            # Log frame info
            if self._frame_count % 30 == 0:  # Log every 30 frames
                drop_info = f" | Dropped: {self._frames_dropped}" if self._frames_dropped > 0 else ""
                logger.info(
                    f"Frame {server_frame} (recv: {self._frame_count}) | "
                    f"Device: {metadata.get('device_id', '?')} | "
                    f"FPS: {self._fps:.1f} | Lag: {self._lag_ms:.0f}ms{drop_info}"
                )
            
            # Call custom callback if provided
            if self.on_frame:
                try:
                    self.on_frame(frame.copy(), metadata)
                except Exception as e:
                    logger.error(f"Error in frame callback: {e}")
            
            # Save frame if save_dir is set
            if self.save_dir:
                self._save_frame(frame, metadata)
            
            # Display frame
            if self.display:
                display_frame = self._prepare_display_frame(frame, metadata)
                window = self.window_name or f"Stream from {metadata.get('device_id', 'unknown')}"
                cv2.imshow(window, display_frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self._running = False
                    break
                elif key == ord('s'):
                    # Manual save on 's' key
                    self._save_frame(frame, metadata, manual=True)
            
            self._frame_count += 1
    
    def _scale_to_fit(self, frame: np.ndarray) -> np.ndarray:
        """
        Scale frame to fit within max display dimensions while maintaining aspect ratio.
        
        Args:
            frame: Original frame
            
        Returns:
            Scaled frame that fits within max_display_width x max_display_height
        """
        h, w = frame.shape[:2]
        
        # Check if scaling is needed
        if w <= self.max_display_width and h <= self.max_display_height:
            return frame
        
        # Calculate scale factor to fit within bounds
        scale_w = self.max_display_width / w
        scale_h = self.max_display_height / h
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize with high quality interpolation
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _prepare_display_frame(self, frame: np.ndarray, metadata: dict) -> np.ndarray:
        """
        Prepare frame for display with overlay information.
        
        Args:
            frame: Original frame
            metadata: Frame metadata
        
        Returns:
            Frame with overlay, scaled to fit display
        """
        # First scale to fit display
        display_frame = self._scale_to_fit(frame)
        
        h, w = display_frame.shape[:2]
        
        # Always show frame number and lag in top-right corner (minimal overlay)
        server_frame = metadata.get('frame_number', '?')
        lag_color = (0, 255, 0) if self._lag_ms < 100 else (0, 165, 255) if self._lag_ms < 200 else (0, 0, 255)
        frame_text = f"F:{server_frame} Lag:{self._lag_ms:.0f}ms"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(frame_text, font, font_scale, thickness)
        cv2.rectangle(display_frame, (w - text_w - 15, 5), (w - 5, text_h + 15), (0, 0, 0), -1)
        cv2.putText(display_frame, frame_text, (w - text_w - 10, text_h + 10), font, font_scale, lag_color, thickness)
        
        # Return early if full overlay is disabled
        if not self.show_overlay:
            return display_frame
        
        # Scale overlay elements based on frame size
        scale_factor = min(w / 640, h / 480)  # Reference size for scaling UI
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, 0.5 * scale_factor)
        color = (0, 255, 0)
        thickness = max(1, int(scale_factor))
        
        # Add semi-transparent background for text
        overlay = display_frame.copy()
        box_width = int(250 * scale_factor)
        box_height = int(110 * scale_factor)
        cv2.rectangle(overlay, (5, 5), (5 + box_width, 5 + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
        
        # Get original resolution for display
        orig_h, orig_w = frame.shape[:2]
        
        y_offset = int(25 * scale_factor)
        line_spacing = int(18 * scale_factor)
        texts = [
            f"Device: {metadata.get('device_id', 'unknown')}",
            f"Frame: {metadata.get('frame_number', '?')}",
            f"Res: {orig_w}x{orig_h} -> {w}x{h}",
            f"Time: {metadata.get('timestamp', '?')[:19]}",
            f"FPS: {self._fps:.1f}"
        ]
        
        for text in texts:
            cv2.putText(display_frame, text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += line_spacing
        
        return display_frame
    
    def _save_frame(self, frame: np.ndarray, metadata: dict, manual: bool = False):
        """
        Save frame to disk.
        
        Args:
            frame: Frame to save
            metadata: Frame metadata
            manual: Whether this is a manual save (uses different naming)
        """
        if manual:
            filename = f"manual_{metadata.get('timestamp', '')[:19].replace(':', '-')}.jpg"
        else:
            filename = f"frame_{metadata.get('frame_number', self._frame_count):06d}.jpg"
        
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, frame)
        
        if manual:
            logger.info(f"Saved frame: {filepath}")
    
    def stop(self):
        """Stop the listener and clean up resources."""
        self._running = False
        self.disconnect()
        
        if self.display:
            cv2.destroyAllWindows()
        
        logger.info(f"Listener stopped. Received {self._frame_count} frames, "
                   f"{self._bytes_received / 1024 / 1024:.2f} MB total")
    
    @property
    def is_running(self) -> bool:
        """Check if the listener is currently running."""
        return self._running
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to a streamer."""
        return self._connected
    
    @property
    def frame_count(self) -> int:
        """Get the number of frames received."""
        return self._frame_count
    
    @property
    def fps(self) -> float:
        """Get the current receive FPS."""
        return self._fps


def main():
    parser = argparse.ArgumentParser(
        description="Image Listener - Receive images from RPi camera streamer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Basic usage:
        python image_listener.py --host 100.64.0.1 --port 5000
    
    Save frames to directory:
        python image_listener.py --host 100.64.0.1 --port 5000 --save ./frames
    
    Headless mode (no display):
        python image_listener.py --host 100.64.0.1 --port 5000 --no-display --save ./frames
    
    With auto-reconnect:
        python image_listener.py --host 100.64.0.1 --port 5000 --reconnect

Controls (when display is enabled):
    q - Quit
    s - Save current frame manually
        """
    )
    
    parser.add_argument("--host", default="100.99.98.1",
                       help="Streamer host address (default: 100.99.98.1)")
    parser.add_argument("--port", type=int, default=5000,
                       help="Streamer port number (default: 5000)")
    parser.add_argument("--save", type=str, default=None,
                       help="Directory to save received frames")
    parser.add_argument("--no-display", action="store_true",
                       help="Disable display window")
    parser.add_argument("--reconnect", action="store_true",
                       help="Auto-reconnect on connection loss")
    parser.add_argument("--reconnect-delay", type=float, default=5.0,
                       help="Delay between reconnection attempts (default: 5.0)")
    parser.add_argument("--window-name", type=str, default=None,
                       help="Custom window name")
    parser.add_argument("--max-width", type=int, default=1280,
                       help="Maximum display width - scales down if larger (default: 1280)")
    parser.add_argument("--max-height", type=int, default=720,
                       help="Maximum display height - scales down if larger (default: 720)")
    parser.add_argument("--show-overlay", action="store_true",
                       help="Show text overlay on display")
    parser.add_argument("--max-lag", type=float, default=200.0,
                       help="Max lag in ms before dropping frames (0 to disable, default: 200)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        experiment_name="listener",
        log_to_file=True,
        log_to_console=True
    )
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start listener
    listener = ImageListener(
        host=args.host,
        port=args.port,
        display=not args.no_display,
        save_dir=args.save,
        window_name=args.window_name,
        max_display_width=args.max_width,
        max_display_height=args.max_height,
        show_overlay=args.show_overlay,
        max_lag_ms=args.max_lag
    )
    
    listener.start(
        auto_reconnect=args.reconnect,
        reconnect_delay=args.reconnect_delay
    )


if __name__ == "__main__":
    main()
