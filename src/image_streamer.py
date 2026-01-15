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


def get_pi_power_stats() -> dict:
    """
    Get Raspberry Pi power statistics using vcgencmd.
    
    Returns dict with:
        - voltage: Core voltage (V)
        - throttled: Throttling status flags
        - temp: CPU temperature (°C)
        - under_voltage: True if under-voltage detected
        - freq_capped: True if frequency capped
        - throttled_now: True if currently throttled
    """
    import subprocess
    
    stats = {
        'voltage': None,
        'temp': None,
        'throttled': None,
        'under_voltage': False,
        'freq_capped': False,
        'throttled_now': False,
        'under_voltage_occurred': False,
    }
    
    try:
        # Get core voltage
        result = subprocess.run(['vcgencmd', 'measure_volts', 'core'], 
                              capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            # Output: volt=1.2000V
            voltage_str = result.stdout.strip().split('=')[1].rstrip('V')
            stats['voltage'] = float(voltage_str)
        
        # Get temperature
        result = subprocess.run(['vcgencmd', 'measure_temp'], 
                              capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            # Output: temp=45.0'C
            temp_str = result.stdout.strip().split('=')[1].rstrip("'C")
            stats['temp'] = float(temp_str)
        
        # Get throttling status
        result = subprocess.run(['vcgencmd', 'get_throttled'], 
                              capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            # Output: throttled=0x0
            throttled_hex = result.stdout.strip().split('=')[1]
            throttled = int(throttled_hex, 16)
            stats['throttled'] = throttled_hex
            
            # Decode throttling flags
            # Bit 0: Under-voltage detected
            # Bit 1: Arm frequency capped
            # Bit 2: Currently throttled
            # Bit 16: Under-voltage has occurred
            # Bit 17: Arm frequency capping has occurred
            # Bit 18: Throttling has occurred
            stats['under_voltage'] = bool(throttled & 0x1)
            stats['freq_capped'] = bool(throttled & 0x2)
            stats['throttled_now'] = bool(throttled & 0x4)
            stats['under_voltage_occurred'] = bool(throttled & 0x10000)
            
    except FileNotFoundError:
        logger.debug("vcgencmd not found - not running on Raspberry Pi?")
    except Exception as e:
        logger.debug(f"Failed to get power stats: {e}")
    
    return stats


def format_power_status(stats: dict) -> str:
    """Format power stats for logging."""
    parts = []
    
    if stats['voltage'] is not None:
        parts.append(f"Voltage: {stats['voltage']:.2f}V")
    
    if stats['temp'] is not None:
        parts.append(f"Temp: {stats['temp']:.1f}°C")
    
    warnings = []
    if stats['under_voltage']:
        warnings.append("⚠️ UNDER-VOLTAGE!")
    if stats['freq_capped']:
        warnings.append("⚠️ FREQ CAPPED!")
    if stats['throttled_now']:
        warnings.append("⚠️ THROTTLED!")
    if stats['under_voltage_occurred']:
        warnings.append("(under-voltage occurred)")
    
    if warnings:
        parts.append(" ".join(warnings))
    
    return " | ".join(parts) if parts else "Power stats unavailable"


# Common sensor resolutions for IMX708 (RPi Camera Module 3 Wide)
IMX708_MODES = {
    "full": (4608, 2592),      # Full resolution (9MP)
    "2x2_binned": (2304, 1296), # 2x2 binned (3MP) - good balance
    "1080p": (1920, 1080),      # 1080p
    "720p": (1280, 720),        # 720p
    "480p": (640, 480),         # VGA
}


class CameraCapture:
    """Handles camera capture on Raspberry Pi using picamera2."""
    
    def __init__(self, resolution: tuple = (640, 480), fps: int = 30, 
                 sensor_mode: str = "full", crop_square: int = 0):
        """
        Initialize camera capture.
        
        Args:
            resolution: Output resolution (width, height)
            fps: Frames per second
            sensor_mode: Sensor mode to use - "full", "binned", or "auto"
                - "full": 4608x2592 sensor readout (max quality, slower)
                - "binned": 2304x1296 2x2 binned readout (good quality, faster)
                - "auto": Let picamera2 choose based on output resolution
            crop_square: If > 0, crop to this square size from center
        """
        self.resolution = resolution
        self.fps = fps
        self.sensor_mode = sensor_mode
        self.crop_square = crop_square  # If > 0, crop to this square size from center
        self.camera = None
        self._running = False
        
    def initialize(self) -> bool:
        """Initialize the camera."""
        try:
            from picamera2 import Picamera2
            
            self.camera = Picamera2()
            
            # Get sensor modes to find the best one for full FOV
            sensor_modes = self.camera.sensor_modes
            logger.info(f"Available sensor modes: {len(sensor_modes)}")
            for i, mode in enumerate(sensor_modes):
                logger.info(f"  Mode {i}: {mode}")
            
            # Configure camera based on sensor_mode setting
            # IMX708 Wide sensor modes:
            #   Mode 0: 1536x864 @ 120fps (cropped center)
            #   Mode 1: 2304x1296 @ 56fps (2x2 binned, full FOV)
            #   Mode 2: 4608x2592 @ 14fps (full resolution, full FOV)
            
            if self.sensor_mode == "full":
                # Use full sensor readout - maximum detail
                sensor_output = IMX708_MODES["full"]  # 4608x2592
                logger.info(f"Using full sensor mode: {sensor_output}")
                config = self.camera.create_preview_configuration(
                    main={"size": self.resolution, "format": "RGB888"},
                    sensor={"output_size": sensor_output, "bit_depth": 10},
                    buffer_count=2
                )
            elif self.sensor_mode == "binned":
                # Use 2x2 binned mode - good quality with better performance
                sensor_output = IMX708_MODES["2x2_binned"]  # 2304x1296
                logger.info(f"Using binned sensor mode: {sensor_output}")
                config = self.camera.create_preview_configuration(
                    main={"size": self.resolution, "format": "RGB888"},
                    sensor={"output_size": sensor_output, "bit_depth": 10},
                    buffer_count=2
                )
            else:
                # Auto mode - let picamera2 choose
                logger.info("Using auto sensor mode selection")
                config = self.camera.create_preview_configuration(
                    main={"size": self.resolution, "format": "RGB888"}
                )
            
            self.camera.configure(config)
            self.camera.start()
            
            # Allow camera to warm up
            time.sleep(2)
            
            # Log actual configuration
            camera_config = self.camera.camera_configuration()
            sensor_size = camera_config.get('sensor', {}).get('output_size', 'unknown')
            main_format = camera_config.get('main', {}).get('format', 'unknown')
            logger.info(f"Sensor mode: {sensor_size}")
            logger.info(f"Main stream format: {main_format}")
            logger.info(f"Camera initialized: {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps (Sensor: {self.sensor_mode})")
            
            # Store the format for color conversion decisions
            self._format = main_format
            
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
        
        # Picamera2 with RGB888 format actually returns BGR order on Pi 5
        # So we DON'T need to convert - just use the frame directly
        # If colors look wrong, uncomment one of these conversions:
        # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # If too blue
        # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # If too red
        frame_bgr = frame  # Try no conversion first
        
        # Crop to square if requested
        if self.crop_square > 0:
            frame_bgr = self._crop_center_square(frame_bgr, self.crop_square)
        
        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded = cv2.imencode('.jpg', frame_bgr, encode_param)
        
        if success:
            return encoded.tobytes()
        return None
    
    def _crop_center_square(self, frame: np.ndarray, size: int) -> np.ndarray:
        """Crop a square from the center of the frame."""
        h, w = frame.shape[:2]
        
        # Calculate crop region (center)
        cx, cy = w // 2, h // 2
        half = size // 2
        
        # Ensure we don't exceed frame bounds
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, x1 + size)
        y2 = min(h, y1 + size)
        
        return frame[y1:y2, x1:x2]
    
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
            self.camera = CameraCapture(
                self.config.resolution, 
                self.config.fps,
                sensor_mode=getattr(self.config, 'sensor_mode', 'full'),
                crop_square=getattr(self.config, 'crop_square', 0)
            )
        
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
        last_power_check = 0
        power_check_interval = 5.0  # Check power every 5 seconds
        
        # Initial power check
        stats = get_pi_power_stats()
        logger.info(f"Power status: {format_power_status(stats)}")
        
        while self._running:
            start_time = time.time()
            
            # Periodic power monitoring
            if start_time - last_power_check >= power_check_interval:
                stats = get_pi_power_stats()
                status_str = format_power_status(stats)
                
                # Always log if there's a warning, otherwise log periodically
                if stats['under_voltage'] or stats['throttled_now'] or stats['freq_capped']:
                    logger.warning(f"Power issue detected! {status_str}")
                else:
                    logger.info(f"Power status: {status_str}")
                
                last_power_check = start_time
            
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
Resolution Modes (IMX708 Wide sensor):
    --full      4608x2592 @ 10fps  (full sensor, max detail, ~14fps max)
    --2500      2500x2500 @ 10fps  (square crop from full sensor)
    --binned    2304x1296 @ 20fps  (2x2 binned sensor, good balance, ~56fps max)
    --1080      1920x1080 @ 30fps  (binned sensor, scaled)
    --720       1280x720  @ 30fps  (binned sensor, scaled)
    --480       640x480   @ 30fps  (auto sensor mode)

Sensor Modes:
    Full (4608x2592):   Max quality, slower, higher power
    Binned (2304x1296): 2x2 pixel binning, faster, lower power, good quality

Examples:
    # Default 1080p streaming (uses binned sensor)
    python image_streamer.py --port 5000
    
    # Full resolution for maximum detail
    python image_streamer.py --full
    
    # 2x2 binned for good quality + performance
    python image_streamer.py --binned
    
    # Custom resolution
    python image_streamer.py --resolution 3840x2160 --fps 5
    
    # Mock camera for testing
    python image_streamer.py --mock --720

Note:
    Both full and binned modes preserve the full wide-angle FOV.
    For receiving images, use image_listener.py on the receiving machine.
        """
    )
    
    parser.add_argument("--host", default="0.0.0.0",
                       help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port number (default: 5000)")
    parser.add_argument("--resolution", type=str, default=None,
                       help="Output resolution WxH (overrides preset modes)")
    parser.add_argument("--fps", type=int, default=None,
                       help="Frames per second (default depends on mode)")
    parser.add_argument("--quality", type=int, default=85,
                       help="JPEG quality 1-100 (default: 85)")
    parser.add_argument("--device-id", type=str, default=None,
                       help="Device identifier (default: hostname)")
    parser.add_argument("--mock", action="store_true",
                       help="Use mock camera for testing")
    
    # Resolution presets (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--full", action="store_true",
                           help="Full resolution 4608x2592 @ 10fps (max detail)")
    mode_group.add_argument("--2500", dest="mode_2500", action="store_true",
                           help="2500x2500 square @ 10fps (cropped from full sensor)")
    mode_group.add_argument("--binned", action="store_true",
                           help="2x2 binned 2304x1296 @ 20fps (good balance)")
    mode_group.add_argument("--1080", "--1080p", dest="mode_1080", action="store_true",
                           help="1920x1080 @ 30fps")
    mode_group.add_argument("--720", "--720p", dest="mode_720", action="store_true",
                           help="1280x720 @ 30fps")
    mode_group.add_argument("--480", "--480p", dest="mode_480", action="store_true",
                           help="640x480 @ 30fps")
    
    args = parser.parse_args()
    
    # Determine resolution, fps, and sensor mode from mode or explicit args
    crop_square = 0  # No square cropping by default
    sensor_mode = "full"  # Default to full sensor readout
    
    if args.full:
        resolution = (4608, 2592)
        sensor_mode = "full"
        default_fps = 10
    elif args.mode_2500:
        resolution = (4608, 2592)  # Capture at full, then crop
        sensor_mode = "full"
        crop_square = 2500
        default_fps = 10
    elif args.binned:
        resolution = (2304, 1296)
        sensor_mode = "binned"  # Use 2x2 binned sensor mode
        default_fps = 20
    elif args.mode_1080:
        resolution = (1920, 1080)
        sensor_mode = "binned"  # Use binned mode, scaled down
        default_fps = 30
    elif args.mode_720:
        resolution = (1280, 720)
        sensor_mode = "binned"  # Use binned mode, scaled down
        default_fps = 30
    elif args.mode_480:
        resolution = (640, 480)
        sensor_mode = "auto"  # Let picamera2 choose
        default_fps = 30
    elif args.resolution:
        try:
            width, height = map(int, args.resolution.split('x'))
            resolution = (width, height)
        except ValueError:
            logger.error(f"Invalid resolution format: {args.resolution}")
            sys.exit(1)
        # For custom resolutions, use binned mode if smaller than full
        sensor_mode = "binned" if resolution[0] <= 2304 else "full"
        default_fps = 30
    else:
        # Default to 1080p with binned sensor
        resolution = (1920, 1080)
        sensor_mode = "binned"
        default_fps = 30
    
    fps = args.fps if args.fps else default_fps
    
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
        fps=fps,
        quality=args.quality
    )
    config.sensor_mode = sensor_mode
    config.crop_square = crop_square
    
    if crop_square > 0:
        logger.info(f"Mode: {resolution[0]}x{resolution[1]} -> {crop_square}x{crop_square} (square crop) @ {fps}fps (sensor: {sensor_mode})")
    else:
        logger.info(f"Mode: {resolution[0]}x{resolution[1]} @ {fps}fps (sensor: {sensor_mode})")
    
    streamer = ImageStreamer(config, device_id=args.device_id, mock=args.mock)
    streamer.start()


if __name__ == "__main__":
    main()
