#!/usr/bin/env python3
"""
Raspberry Pi Camera Image Streamer

Captures images from the Raspberry Pi camera and broadcasts them over
a Tailscale network to connected listeners. Each frame includes a timestamp
and device metadata.

This script is designed to run on the Raspberry Pi.
For the receiver/listener, use image_listener.py instead.

Usage:
    python image_streamer.py --port 5000
    python image_streamer.py --port 5000 --resolution 1280x720 --fps 15
    python image_streamer.py --port 5000 --mock  # For testing without camera

Requirements:
    - picamera2 (for Raspberry Pi camera)
    - opencv-python (for image encoding)
    - numpy
"""

import argparse
import logging
import signal
import sys
import socket
import threading
import time
from typing import Optional, List

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from utils.common import (
    create_metadata,
    send_frame,
    get_tailscale_ip,
    get_hostname,
    StreamConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CameraCapture:
    """Handles camera capture on Raspberry Pi using picamera2."""
    
    def __init__(self, resolution: tuple = (640, 480), fps: int = 30):
        self.resolution = resolution
        self.fps = fps
        self.camera = None
        self._running = False
        
    def initialize(self) -> bool:
        """Initialize the camera."""
        try:
            from picamera2 import Picamera2
            
            self.camera = Picamera2()
            
            # Configure camera
            config = self.camera.create_preview_configuration(
                main={"size": self.resolution, "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            
            # Allow camera to warm up
            time.sleep(2)
            
            logger.info(f"Camera initialized: {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps")
            self._running = True
            return True
            
        except ImportError:
            logger.error("picamera2 not installed. Install with: sudo apt install python3-picamera2")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the camera."""
        if self.camera is None or not self._running:
            return None
        
        try:
            frame = self.camera.capture_array()
            return frame
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None
    
    def capture_jpeg(self, quality: int = 85) -> Optional[bytes]:
        """Capture a frame and encode as JPEG."""
        frame = self.capture_frame()
        if frame is None:
            return None
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded = cv2.imencode('.jpg', frame_bgr, encode_param)
        
        if success:
            return encoded.tobytes()
        return None
    
    def close(self):
        """Close the camera."""
        self._running = False
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
            except Exception:
                pass
            self.camera = None
        logger.info("Camera closed")


class MockCameraCapture:
    """Mock camera for testing without actual hardware."""
    
    def __init__(self, resolution: tuple = (640, 480), fps: int = 30):
        self.resolution = resolution
        self.fps = fps
        self._running = False
        self._frame_count = 0
    
    def initialize(self) -> bool:
        self._running = True
        logger.info(f"Mock camera initialized: {self.resolution[0]}x{self.resolution[1]}")
        return True
    
    def capture_frame(self) -> Optional[np.ndarray]:
        if not self._running:
            return None
        
        # Generate a test pattern
        frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # Add some visual elements
        cv2.putText(frame, f"Frame: {self._frame_count}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(frame, (self.resolution[0]//2, self.resolution[1]//2), 
                  50 + (self._frame_count % 50), (0, 0, 255), 3)
        
        self._frame_count += 1
        return frame
    
    def capture_jpeg(self, quality: int = 85) -> Optional[bytes]:
        frame = self.capture_frame()
        if frame is None:
            return None
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded = cv2.imencode('.jpg', frame, encode_param)
        
        if success:
            return encoded.tobytes()
        return None
    
    def close(self):
        self._running = False
        logger.info("Mock camera closed")


class ImageStreamer:
    """Streams images from the camera to connected clients."""
    
    def __init__(self, config: StreamConfig, device_id: str = None, mock: bool = False):
        self.config = config
        self.device_id = device_id or get_hostname()
        self.mock = mock
        
        self.camera = None
        self.server_socket = None
        self.clients: List[socket.socket] = []
        self.clients_lock = threading.Lock()
        self._running = False
        self._frame_count = 0
        
    def start(self):
        """Start the image streamer."""
        # Initialize camera
        if self.mock:
            self.camera = MockCameraCapture(self.config.resolution, self.config.fps)
        else:
            self.camera = CameraCapture(self.config.resolution, self.config.fps)
        
        if not self.camera.initialize():
            logger.error("Failed to initialize camera")
            return False
        
        # Create server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind((self.config.host, self.config.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)
        except Exception as e:
            logger.error(f"Failed to bind server socket: {e}")
            return False
        
        self._running = True
        
        # Start accept thread
        accept_thread = threading.Thread(target=self._accept_clients, daemon=True)
        accept_thread.start()
        
        tailscale_ip = get_tailscale_ip()
        logger.info(f"Image streamer started on {self.config.host}:{self.config.port}")
        logger.info(f"Tailscale IP: {tailscale_ip}")
        logger.info(f"Listeners can connect to: {tailscale_ip}:{self.config.port}")
        
        # Main streaming loop
        self._stream_loop()
        
        return True
    
    def _accept_clients(self):
        """Accept incoming client connections."""
        while self._running:
            try:
                client_socket, addr = self.server_socket.accept()
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                with self.clients_lock:
                    self.clients.append(client_socket)
                
                logger.info(f"Client connected from {addr}")
                
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Error accepting client: {e}")
    
    def _stream_loop(self):
        """Main loop for capturing and streaming frames."""
        frame_interval = 1.0 / self.config.fps
        
        while self._running:
            start_time = time.time()
            
            # Capture frame
            image_data = self.camera.capture_jpeg(self.config.quality)
            if image_data is None:
                continue
            
            # Create metadata
            metadata = create_metadata(
                device_id=self.device_id,
                additional_data={
                    "frame_number": self._frame_count,
                    "resolution": self.config.resolution,
                    "quality": self.config.quality,
                    "format": self.config.format
                }
            )
            
            # Send to all clients
            self._broadcast_frame(image_data, metadata)
            
            self._frame_count += 1
            
            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _broadcast_frame(self, image_data: bytes, metadata: dict):
        """Broadcast frame to all connected clients."""
        disconnected = []
        
        with self.clients_lock:
            for client in self.clients:
                try:
                    send_frame(client, image_data, metadata)
                except Exception as e:
                    logger.warning(f"Client disconnected: {e}")
                    disconnected.append(client)
            
            # Remove disconnected clients
            for client in disconnected:
                self.clients.remove(client)
                try:
                    client.close()
                except Exception:
                    pass
    
    def stop(self):
        """Stop the streamer."""
        self._running = False
        
        # Close all clients
        with self.clients_lock:
            for client in self.clients:
                try:
                    client.close()
                except Exception:
                    pass
            self.clients.clear()
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        
        # Close camera
        if self.camera:
            self.camera.close()
        
        logger.info("Image streamer stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi Camera Image Streamer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Start streamer on Raspberry Pi:
        python image_streamer.py --port 5000
    
    Start streamer with custom resolution:
        python image_streamer.py --port 5000 --resolution 1280x720 --fps 15
    
    Start streamer with mock camera (for testing):
        python image_streamer.py --port 5000 --mock

Note:
    For receiving images, use image_listener.py on the receiving machine.
        """
    )
    
    parser.add_argument("--host", default="0.0.0.0",
                       help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port number (default: 5000)")
    parser.add_argument("--resolution", type=str, default="640x480",
                       help="Camera resolution WxH (default: 640x480)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second (default: 30)")
    parser.add_argument("--quality", type=int, default=85,
                       help="JPEG quality 1-100 (default: 85)")
    parser.add_argument("--device-id", type=str, default=None,
                       help="Device identifier (default: hostname)")
    parser.add_argument("--mock", action="store_true",
                       help="Use mock camera for testing")
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        logger.error(f"Invalid resolution format: {args.resolution}")
        sys.exit(1)
    
    # Setup signal handlers
    streamer = None
    
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        if streamer:
            streamer.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start streamer
    config = StreamConfig(
        host=args.host,
        port=args.port,
        resolution=resolution,
        fps=args.fps,
        quality=args.quality
    )
    
    streamer = ImageStreamer(config, device_id=args.device_id, mock=args.mock)
    streamer.start()


if __name__ == "__main__":
    main()
