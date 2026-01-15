#!/usr/bin/env python3
"""
Image Collector for YOLO Training Data

Collects images from the RPi camera stream via ImageListener,
or from local camera/video files for building TurtleBot detection 
training datasets.
"""

import logging
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not installed. Install with: pip install opencv-python")

# Import ImageListener for RPi stream
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from src.image_listener import ImageListener
    LISTENER_AVAILABLE = True
except ImportError:
    LISTENER_AVAILABLE = False
    logger.warning("ImageListener not available")


@dataclass
class CollectorConfig:
    """Configuration for image collection."""
    output_dir: str = "collected_images"
    image_format: str = "jpg"  # jpg, png
    jpeg_quality: int = 95
    resize: Optional[Tuple[int, int]] = None  # (width, height) or None
    prefix: str = "img"
    auto_capture_interval: float = 0.0  # seconds (0 = manual only)
    max_images: int = 0  # 0 = unlimited


class ImageCollector:
    """
    Collects images from camera or video for training data.
    
    Supports:
    - Live camera capture
    - Video file frame extraction  
    - Network stream capture (RTSP, HTTP)
    - Manual and automatic capture modes
    
    Controls (live mode):
        - Space: Capture image
        - 'a': Toggle auto-capture
        - '+'/'-': Adjust auto-capture interval
        - 'q': Quit
    
    Example:
        >>> collector = ImageCollector(source=0)  # Webcam
        >>> collector.collect_interactive()
        
        >>> # Or from video file
        >>> collector = ImageCollector(source="video.mp4")
        >>> collector.extract_frames(every_n=30)  # Every 30 frames
    """
    
    def __init__(
        self,
        source: any = 0,
        config: Optional[CollectorConfig] = None
    ):
        """
        Initialize image collector.
        
        Args:
            source: Camera index (int), video path (str), or URL (str)
            config: Collection configuration
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required. Install with: pip install opencv-python")
        
        self.source = source
        self.config = config or CollectorConfig()
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.captured_count = 0
        self.auto_capture = False
        self.last_capture_time = 0.0
    
    def _open_source(self) -> bool:
        """Open the video source."""
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open source: {self.source}")
            return False
        
        # Get source info
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        logger.info(f"Opened source: {self.source} ({self.frame_width}x{self.frame_height} @ {self.fps:.1f}fps)")
        return True
    
    def _close_source(self) -> None:
        """Close the video source."""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def _get_filename(self) -> str:
        """Generate filename for captured image."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{self.config.prefix}_{timestamp}.{self.config.image_format}"
        return str(self.output_dir / filename)
    
    def _save_image(self, frame: np.ndarray) -> str:
        """Save image to file."""
        # Resize if configured
        if self.config.resize:
            frame = cv2.resize(frame, self.config.resize)
        
        filepath = self._get_filename()
        
        # Save with appropriate quality settings
        if self.config.image_format.lower() == "jpg":
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
        else:
            cv2.imwrite(filepath, frame)
        
        self.captured_count += 1
        logger.info(f"Captured: {filepath} ({self.captured_count} total)")
        
        return filepath
    
    def capture_single(self) -> Optional[str]:
        """
        Capture a single image from the source.
        
        Returns:
            Path to saved image or None if failed
        """
        if not self._open_source():
            return None
        
        ret, frame = self.cap.read()
        self._close_source()
        
        if not ret:
            logger.error("Failed to capture frame")
            return None
        
        return self._save_image(frame)
    
    def collect_interactive(self, window_name: str = "Image Collector") -> int:
        """
        Interactive collection mode with live preview.
        
        Args:
            window_name: OpenCV window name
            
        Returns:
            Number of images captured
        """
        if not self._open_source():
            return 0
        
        print("\n=== Image Collector ===")
        print("Controls:")
        print("  Space: Capture image")
        print("  'a': Toggle auto-capture")
        print("  '+'/'-': Adjust interval")
        print("  'q': Quit")
        print("=" * 25 + "\n")
        
        auto_interval = self.config.auto_capture_interval
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break
            
            # Check max images
            if self.config.max_images > 0 and self.captured_count >= self.config.max_images:
                logger.info(f"Reached max images: {self.config.max_images}")
                break
            
            # Auto capture
            current_time = time.time()
            if self.auto_capture and auto_interval > 0:
                if current_time - self.last_capture_time >= auto_interval:
                    self._save_image(frame)
                    self.last_capture_time = current_time
            
            # Draw info overlay
            display = frame.copy()
            self._draw_overlay(display, auto_interval)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == 32:  # Space
                self._save_image(frame)
            elif key == ord('a'):
                self.auto_capture = not self.auto_capture
                if self.auto_capture:
                    self.last_capture_time = time.time()
            elif key == ord('+') or key == ord('='):
                auto_interval = min(auto_interval + 0.5, 10.0)
            elif key == ord('-'):
                auto_interval = max(auto_interval - 0.5, 0.5)
        
        self._close_source()
        cv2.destroyAllWindows()
        
        logger.info(f"Collection complete: {self.captured_count} images")
        return self.captured_count
    
    def _draw_overlay(self, frame: np.ndarray, auto_interval: float) -> None:
        """Draw info overlay on frame."""
        h, w = frame.shape[:2]
        
        # Background bar
        cv2.rectangle(frame, (0, 0), (w, 35), (0, 0, 0), -1)
        
        # Info text
        auto_str = f"AUTO: {auto_interval:.1f}s" if self.auto_capture else "AUTO: OFF"
        info = f"Captured: {self.captured_count} | {auto_str} | Press SPACE to capture"
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Recording indicator
        if self.auto_capture:
            cv2.circle(frame, (w - 25, 18), 10, (0, 0, 255), -1)
    
    def extract_frames(
        self,
        every_n: int = 30,
        max_frames: int = 0,
        skip_start: int = 0,
        skip_end: int = 0
    ) -> int:
        """
        Extract frames from video file.
        
        Args:
            every_n: Extract every N frames
            max_frames: Maximum frames to extract (0 = all)
            skip_start: Skip first N frames
            skip_end: Skip last N frames
            
        Returns:
            Number of frames extracted
        """
        if not self._open_source():
            return 0
        
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            logger.warning("Cannot determine total frames, extracting until end")
            total_frames = float('inf')
        
        end_frame = total_frames - skip_end
        frame_idx = 0
        extracted = 0
        
        # Skip start frames
        if skip_start > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, skip_start)
            frame_idx = skip_start
        
        logger.info(f"Extracting frames from {skip_start} to {end_frame}, every {every_n}")
        
        while frame_idx < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_idx % every_n == 0:
                self._save_image(frame)
                extracted += 1
                
                if max_frames > 0 and extracted >= max_frames:
                    break
            
            frame_idx += 1
            
            # Progress update
            if frame_idx % 500 == 0:
                logger.info(f"Progress: frame {frame_idx}, extracted {extracted}")
        
        self._close_source()
        
        logger.info(f"Extraction complete: {extracted} frames from {frame_idx} total")
        return extracted
    
    def collect_from_stream(
        self,
        duration: float = 60.0,
        interval: float = 1.0
    ) -> int:
        """
        Collect images from a stream for a duration.
        
        Args:
            duration: Collection duration in seconds
            interval: Time between captures in seconds
            
        Returns:
            Number of images captured
        """
        if not self._open_source():
            return 0
        
        start_time = time.time()
        last_capture = 0.0
        
        logger.info(f"Collecting from stream for {duration}s at {interval}s intervals")
        
        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Stream read failed, retrying...")
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            if current_time - last_capture >= interval:
                self._save_image(frame)
                last_capture = current_time
            
            # Check max images
            if self.config.max_images > 0 and self.captured_count >= self.config.max_images:
                break
            
            time.sleep(0.01)  # Small delay to prevent CPU overuse
        
        self._close_source()
        
        logger.info(f"Stream collection complete: {self.captured_count} images")
        return self.captured_count


class StreamCollector:
    """
    Collects images from RPi camera stream via ImageListener.
    
    This collector connects to an ImageStreamer running on a Raspberry Pi
    and saves frames for training data collection.
    
    Controls (interactive mode):
        - Space: Capture current frame
        - 'a': Toggle auto-capture
        - '+'/'-': Adjust auto-capture interval
        - 'q': Quit
    
    Example:
        >>> collector = StreamCollector(host="100.99.98.1", port=5000)
        >>> collector.collect_interactive()
        
        >>> # Or auto-collect for a duration
        >>> collector.collect_timed(duration=60, interval=2.0)
    """
    
    def __init__(
        self,
        host: str = "100.99.98.1",
        port: int = 5000,
        config: Optional[CollectorConfig] = None
    ):
        """
        Initialize stream collector.
        
        Args:
            host: RPi Tailscale IP address
            port: ImageStreamer port
            config: Collection configuration
        """
        if not LISTENER_AVAILABLE:
            raise ImportError(
                "ImageListener not available. Ensure rpi_loc.src.image_listener is accessible."
            )
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required. Install with: pip install opencv-python")
        
        self.host = host
        self.port = port
        self.config = config or CollectorConfig()
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.captured_count = 0
        self.auto_capture = False
        self.last_capture_time = 0.0
        
        # Current frame (updated by listener callback)
        self._current_frame: Optional[np.ndarray] = None
        self._current_metadata: Optional[Dict[str, Any]] = None
        self._frame_lock = threading.Lock()
        self._running = False
        
        # Listener
        self._listener: Optional[ImageListener] = None
    
    def _get_filename(self) -> str:
        """Generate filename for captured image."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{self.config.prefix}_{timestamp}.{self.config.image_format}"
        return str(self.output_dir / filename)
    
    def _save_image(self, frame: np.ndarray) -> str:
        """Save image to file."""
        if self.config.resize:
            frame = cv2.resize(frame, self.config.resize)
        
        filepath = self._get_filename()
        
        if self.config.image_format.lower() == "jpg":
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
        else:
            cv2.imwrite(filepath, frame)
        
        self.captured_count += 1
        logger.info(f"Captured: {filepath} ({self.captured_count} total)")
        
        return filepath
    
    def _on_frame(self, frame: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Callback for received frames."""
        with self._frame_lock:
            self._current_frame = frame.copy()
            self._current_metadata = metadata.copy() if metadata else {}
    
    def _start_listener(self) -> bool:
        """Start the image listener in background."""
        self._listener = ImageListener(
            host=self.host,
            port=self.port,
            display=False,  # We handle display ourselves
            on_frame=self._on_frame
        )
        
        if not self._listener.connect():
            logger.error("Failed to connect to streamer")
            return False
        
        # Start listener in background thread
        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listener.start,
            kwargs={"auto_reconnect": True},
            daemon=True
        )
        self._listener_thread.start()
        
        # Wait for first frame
        timeout = 5.0
        start = time.time()
        while time.time() - start < timeout:
            with self._frame_lock:
                if self._current_frame is not None:
                    logger.info("Connected and receiving frames")
                    return True
            time.sleep(0.1)
        
        logger.error("Timeout waiting for first frame")
        return False
    
    def _stop_listener(self) -> None:
        """Stop the image listener."""
        self._running = False
        if self._listener:
            self._listener.stop()
            self._listener = None
    
    def collect_interactive(self, window_name: str = "Stream Collector") -> int:
        """
        Interactive collection mode with live preview from RPi stream.
        
        Args:
            window_name: OpenCV window name
            
        Returns:
            Number of images captured
        """
        if not self._start_listener():
            return 0
        
        print("\n=== Stream Collector (RPi) ===")
        print(f"Connected to {self.host}:{self.port}")
        print("Controls:")
        print("  Space: Capture image")
        print("  'a': Toggle auto-capture")
        print("  '+'/'-': Adjust interval")
        print("  'q': Quit")
        print("=" * 30 + "\n")
        
        auto_interval = self.config.auto_capture_interval or 2.0
        
        try:
            while self._running:
                # Get current frame
                with self._frame_lock:
                    if self._current_frame is None:
                        time.sleep(0.01)
                        continue
                    frame = self._current_frame.copy()
                    metadata = self._current_metadata.copy() if self._current_metadata else {}
                
                # Check max images
                if self.config.max_images > 0 and self.captured_count >= self.config.max_images:
                    logger.info(f"Reached max images: {self.config.max_images}")
                    break
                
                # Auto capture
                current_time = time.time()
                if self.auto_capture and auto_interval > 0:
                    if current_time - self.last_capture_time >= auto_interval:
                        self._save_image(frame)
                        self.last_capture_time = current_time
                
                # Draw info overlay
                display = frame.copy()
                self._draw_overlay(display, auto_interval, metadata)
                
                cv2.imshow(window_name, display)
                
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == 32:  # Space
                    self._save_image(frame)
                elif key == ord('a'):
                    self.auto_capture = not self.auto_capture
                    if self.auto_capture:
                        self.last_capture_time = time.time()
                    print(f"Auto-capture: {'ON' if self.auto_capture else 'OFF'}")
                elif key == ord('+') or key == ord('='):
                    auto_interval = min(auto_interval + 0.5, 10.0)
                    print(f"Interval: {auto_interval:.1f}s")
                elif key == ord('-'):
                    auto_interval = max(auto_interval - 0.5, 0.5)
                    print(f"Interval: {auto_interval:.1f}s")
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._stop_listener()
            cv2.destroyAllWindows()
        
        logger.info(f"Collection complete: {self.captured_count} images")
        return self.captured_count
    
    def _draw_overlay(
        self, 
        frame: np.ndarray, 
        auto_interval: float,
        metadata: Dict[str, Any]
    ) -> None:
        """Draw info overlay on frame."""
        h, w = frame.shape[:2]
        
        # Background bar
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        
        # Info text
        auto_str = f"AUTO: {auto_interval:.1f}s" if self.auto_capture else "AUTO: OFF"
        ts = metadata.get("timestamp", "")
        if ts:
            ts_str = f" | TS: {ts:.3f}" if isinstance(ts, float) else f" | {ts}"
        else:
            ts_str = ""
        
        info = f"Captured: {self.captured_count} | {auto_str}{ts_str}"
        cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Recording indicator
        if self.auto_capture:
            cv2.circle(frame, (w - 25, 20), 10, (0, 0, 255), -1)
        
        # Connection info at bottom
        cv2.rectangle(frame, (0, h - 25), (w, h), (0, 0, 0), -1)
        conn_info = f"Stream: {self.host}:{self.port} | Press SPACE to capture"
        cv2.putText(frame, conn_info, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def collect_timed(
        self,
        duration: float = 60.0,
        interval: float = 2.0,
        show_preview: bool = False
    ) -> int:
        """
        Collect images for a specified duration.
        
        Args:
            duration: Collection duration in seconds
            interval: Time between captures in seconds
            show_preview: Whether to show live preview window
            
        Returns:
            Number of images captured
        """
        if not self._start_listener():
            return 0
        
        logger.info(f"Collecting from stream for {duration}s at {interval}s intervals")
        
        start_time = time.time()
        last_capture = 0.0
        
        try:
            while time.time() - start_time < duration and self._running:
                with self._frame_lock:
                    if self._current_frame is None:
                        time.sleep(0.01)
                        continue
                    frame = self._current_frame.copy()
                
                current_time = time.time()
                if current_time - last_capture >= interval:
                    self._save_image(frame)
                    last_capture = current_time
                
                # Check max images
                if self.config.max_images > 0 and self.captured_count >= self.config.max_images:
                    break
                
                # Optional preview
                if show_preview:
                    cv2.imshow("Collecting...", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._stop_listener()
            if show_preview:
                cv2.destroyAllWindows()
        
        logger.info(f"Timed collection complete: {self.captured_count} images")
        return self.captured_count


def collect_from_camera(
    output_dir: str = "collected_images",
    camera_id: int = 0
) -> int:
    """
    Quick function to collect images from camera.
    
    Args:
        output_dir: Directory to save images
        camera_id: Camera device ID
        
    Returns:
        Number of images captured
    """
    config = CollectorConfig(output_dir=output_dir)
    collector = ImageCollector(source=camera_id, config=config)
    return collector.collect_interactive()


def extract_from_video(
    video_path: str,
    output_dir: str = "extracted_frames",
    every_n: int = 30
) -> int:
    """
    Quick function to extract frames from video.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        every_n: Extract every N frames
        
    Returns:
        Number of frames extracted
    """
    config = CollectorConfig(output_dir=output_dir)
    collector = ImageCollector(source=video_path, config=config)
    return collector.extract_frames(every_n=every_n)


def collect_from_rpi(
    host: str = "100.99.98.1",
    port: int = 5000,
    output_dir: str = "collected_images",
    duration: Optional[float] = None,
    interval: float = 2.0
) -> int:
    """
    Collect images from RPi camera stream.
    
    Args:
        host: RPi Tailscale IP address
        port: ImageStreamer port
        output_dir: Directory to save images
        duration: Collection duration in seconds (None = interactive mode)
        interval: Auto-capture interval in seconds
        
    Returns:
        Number of images captured
    """
    config = CollectorConfig(
        output_dir=output_dir,
        auto_capture_interval=interval
    )
    collector = StreamCollector(host=host, port=port, config=config)
    
    if duration is not None:
        return collector.collect_timed(duration=duration, interval=interval)
    else:
        return collector.collect_interactive()
