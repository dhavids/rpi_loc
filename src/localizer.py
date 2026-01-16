#!/usr/bin/env python3
"""
TurtleBot Localizer

Real-time TurtleBot localization using:
- Image streaming from Raspberry Pi camera
- ArUco reference markers for world coordinate calibration
- YOLO object detection for TurtleBot tracking
- Homography-based transformation to world coordinates
- Position broadcasting and CSV logging

Usage:
    python localizer.py --host 100.99.98.1 --port 5000
    python localizer.py --config config.json --broadcast --csv positions.csv

The localizer:
1. Connects to the RPi camera stream
2. Detects ArUco reference markers (required for calibration)
3. Loads YOLO model for TurtleBot detection
4. Tracks TurtleBot positions with consistent local IDs
5. Transforms positions to world coordinates using homography
6. Broadcasts positions and/or logs to CSV
"""

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Callable

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image_listener import ImageListener
from utils.core.broadcaster import PositionBroadcaster
from utils.core.csv_writer import PositionCSVWriter
from utils.core.sink import (
    PositionSink,
    TurtleBotPosition,
    ReferenceMarkerPosition
)
from utils.core.position_tracker import PositionTracker, TrackedObject
from utils.core.setup_logging import setup_logging, get_named_logger
from utils.core.plotter import PositionPlotter

# Configure logging (will be called from main, but set up default logger here)
logger = get_named_logger("localizer", __name__)


# Default paths relative to rpi_loc root
_THIS_DIR = Path(__file__).parent
_RPI_LOC_ROOT = _THIS_DIR.parent
_DEFAULT_MODEL_PATH = _RPI_LOC_ROOT / "files" / "models" / "yolo" / "runs" / "detect" / "runs" / "train" / "turtlebot" / "weights" / "best.pt"
_DEFAULT_CSV_DIR = _RPI_LOC_ROOT / "files" / "logs"
_DEFAULT_CAPTURE_DIR = _RPI_LOC_ROOT / "files" / "localizer"


class ReferenceMarkerError(Exception):
    """Raised when reference markers are not detected."""
    pass


class StreamTimeoutError(Exception):
    """Raised when no frames are received for too long."""
    pass


@dataclass
class MarkerConfig:
    """Configuration for a reference ArUco marker."""
    marker_id: int
    world_x: float
    world_y: float
    r_index: int = 0  # R0, R1, R2, R3 index for broadcast naming


@dataclass
class LocalizerConfig:
    """Configuration for the localizer."""
    # Stream settings
    host: str = "100.99.98.1"
    port: int = 5000
    
    # Reference markers (ArUco)
    reference_markers: List[MarkerConfig] = None
    aruco_dict: int = cv2.aruco.DICT_4X4_50
    min_reference_markers: int = 4  # Minimum markers needed for homography
    
    # YOLO model
    model_path: Optional[str] = None
    confidence_threshold: float = 0.5
    
    # Output
    broadcast_port: int = 5555
    broadcast_rate_hz: float = 20.0
    csv_rate_hz: float = 10.0
    
    # Timeouts
    stream_timeout: float = 3.0  # Stop broadcasting if no frames for this long
    
    # Display
    display: bool = True
    overlay_detections: bool = False  # Show YOLO bounding boxes on display
    
    # Capture
    capture_images: bool = False  # Periodically save raw frames
    capture_interval: float = 5.0  # Seconds between captures
    capture_dir: Optional[str] = None  # Directory to save captures
    
    def __post_init__(self):
        # Default reference markers (4m x 4m area) with TR as (0,0) and BL as (4,4)
        # R-index order: R0=(0,0), R1=(0,3.5), R2=(0,4), R3=(4,3.5)
        if self.reference_markers is None:
            self.reference_markers = [
                MarkerConfig(marker_id=1, world_x=0.0, world_y=0.0, r_index=0),  # R0 - TR
                MarkerConfig(marker_id=0, world_x=4.0, world_y=0.0, r_index=1),  # R1 - TL (moved)
                MarkerConfig(marker_id=3, world_x=0.0, world_y=3.5, r_index=2),  # R2 - BR
                MarkerConfig(marker_id=2, world_x=4.0, world_y=3.5, r_index=3),  # R3 - BL
            ]


class ArucoDetector:
    """Handles ArUco marker detection."""
    
    def __init__(self, dictionary_type: int = cv2.aruco.DICT_4X4_50):
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.parameters = cv2.aruco.DetectorParameters()
        
        # Tune parameters for better detection in poor lighting conditions
        # Adaptive thresholding - use smaller window for local contrast
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        self.parameters.adaptiveThreshConstant = 7
        
        # Reduce minimum marker perimeter to detect smaller/distant markers
        self.parameters.minMarkerPerimeterRate = 0.01  # Default: 0.03
        self.parameters.maxMarkerPerimeterRate = 4.0   # Default: 4.0
        
        # Be more lenient with marker shape
        self.parameters.polygonalApproxAccuracyRate = 0.05  # Default: 0.03
        self.parameters.minCornerDistanceRate = 0.01  # Default: 0.05
        
        # Corner refinement for better accuracy
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        self.parameters.cornerRefinementMaxIterations = 50
        self.parameters.cornerRefinementMinAccuracy = 0.1
        
        # Error correction - allow more bit errors for damaged/occluded markers
        self.parameters.errorCorrectionRate = 0.8  # Default: 0.6
        
        # Perspective removal - helps with angled markers
        self.parameters.perspectiveRemovePixelPerCell = 8  # Default: 4
        self.parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
        
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        logger.info("ArUco detector initialized with enhanced low-light parameters")
    
    def detect(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Detect ArUco markers in frame.
        
        Returns:
            Dictionary mapping marker_id to corner array
        """
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return {}
        return {int(ids[i][0]): corners[i] for i in range(len(ids))}
    
    def get_center(self, corners: np.ndarray) -> tuple:
        """Get center point of marker from its corners."""
        center = corners.mean(axis=1).flatten()
        return (center[0], center[1])


class YOLODetector:
    """YOLO-based TurtleBot detector."""
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(str(model_path))
        self.confidence_threshold = confidence_threshold
        self.class_names = self.model.names
        logger.info(f"Model loaded with classes: {self.class_names}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect TurtleBots in frame.
        
        Returns:
            List of detections with center, bbox, confidence, class_id
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                detections.append({
                    "pixel_x": cx,
                    "pixel_y": cy,
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                    "confidence": float(boxes.conf[i].cpu().numpy()),
                    "class_id": int(boxes.cls[i].cpu().numpy())
                })
        
        return detections


class TurtleBotLocalizer:
    """
    Real-time TurtleBot localization system.
    
    Combines image streaming, ArUco detection, YOLO detection,
    position tracking, and data output (broadcast/CSV).
    """
    
    def __init__(
        self,
        config: LocalizerConfig,
        csv_path: Optional[str] = None,
        broadcast: bool = False,
        plot: bool = False,
        on_pose_update: Optional[Callable] = None
    ):
        """
        Initialize the localizer.
        
        Args:
            config: Localizer configuration
            csv_path: Path to save CSV positions (None to disable)
            broadcast: Whether to broadcast positions
            plot: Whether to enable position plotter
            on_pose_update: Callback for pose updates
        """
        self.config = config
        self.on_pose_update = on_pose_update
        
        # Initialize ArUco detector
        self.aruco_detector = ArucoDetector(config.aruco_dict)
        
        # Build reference marker lookup (keep full MarkerConfig objects)
        self.reference_markers = {
            m.marker_id: m
            for m in config.reference_markers
        }
        
        # YOLO detector (initialized when reference markers detected)
        self._yolo_detector: Optional[YOLODetector] = None
        self._model_path = config.model_path or str(_DEFAULT_MODEL_PATH)
        
        # Position tracker
        self.tracker = PositionTracker(
            use_smoothing=True,
            use_pixel_association=True  # Use pixels until homography stable
        )
        
        # Homography
        self._homography: Optional[np.ndarray] = None
        self._homography_valid = False
        self._calibrated = False
        
        # Initialize outputs
        self._csv_writer: Optional[PositionCSVWriter] = None
        self._broadcaster: Optional[PositionBroadcaster] = None
        self._sink: Optional[PositionSink] = None
        
        if csv_path:
            csv_path = Path(csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_writer = PositionCSVWriter(csv_path, rate_hz=config.csv_rate_hz)
        
        if broadcast:
            self._broadcaster = PositionBroadcaster(
                port=config.broadcast_port,
                rate_hz=config.broadcast_rate_hz
            )
        
        # Plotter
        self._plotter: Optional[PositionPlotter] = None
        if plot:
            self._plotter = PositionPlotter(
                decay_seconds=15.0,
                trail_seconds=10.0,
                refresh_interval=0.2,
                velocity_scale=0.5,
                show_pos=True
            )
        
        if self._csv_writer or self._broadcaster:
            self._sink = PositionSink(
                csv_writer=self._csv_writer,
                broadcaster=self._broadcaster
            )
        
        # Image listener
        self.listener = ImageListener(
            host=config.host,
            port=config.port,
            display=False,  # We handle display ourselves
            on_frame=self._on_frame
        )
        
        # State
        self._running = False
        self._last_frame_time = 0.0
        self._frame_count = 0
        self._paused = False
        self._last_yolo_detections: List[Dict] = []  # Store for overlay drawing (only valid detections)
        
        # Capture state
        self._capture_enabled = config.capture_images
        self._capture_interval = config.capture_interval
        self._capture_dir = Path(config.capture_dir) if config.capture_dir else _DEFAULT_CAPTURE_DIR
        self._last_capture_time = 0.0
        self._capture_count = 0
        
        if self._capture_enabled:
            self._capture_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Image capture enabled: saving to {self._capture_dir} every {self._capture_interval}s")
        
        # Display
        self._window_name = "TurtleBot Localizer"
        self._screen_size = self._get_screen_size()
        self._max_display_width = int(self._screen_size[0] * 0.9)  # 90% of screen
        self._max_display_height = int(self._screen_size[1] * 0.85)  # 85% of screen (leave room for taskbar)
        
        logger.info(f"Localizer initialized - connecting to {config.host}:{config.port}")
        logger.info(f"Reference markers: {list(self.reference_markers.keys())}")
        if config.display:
            logger.info(f"Screen size: {self._screen_size}, max display: {self._max_display_width}x{self._max_display_height}")
    
    def _on_frame(self, frame: np.ndarray, metadata: Dict):
        """Process each received frame."""
        timestamp = metadata.get('timestamp_ms', time.time() * 1000) / 1000.0
        frame_number = metadata.get('frame_number', 0)
        
        self._last_frame_time = time.time()
        self._frame_count += 1
        
        # Resume if we were paused
        if self._paused:
            logger.info("Stream resumed - resuming broadcast/logging")
            self._paused = False
            if self._sink:
                self._sink.resume()
        
        # Detect ArUco reference markers
        markers = self.aruco_detector.detect(frame)
        ref_markers_found = [mid for mid in markers if mid in self.reference_markers]
        
        # Update homography if enough markers
        if len(ref_markers_found) >= self.config.min_reference_markers:
            self._update_homography(markers)
            
            # First time we have valid calibration
            if not self._calibrated and self._homography_valid:
                logger.info("Calibration complete - all reference markers detected")
                self._calibrated = True
                
                # If plotter is enabled, disable display window
                if self._plotter:
                    self.config.display = False
                    cv2.destroyAllWindows()
                    logger.info("Display disabled - using plotter instead")
                
                # Load YOLO model now
                if self._yolo_detector is None:
                    self._load_yolo_model()
        
        elif not self._calibrated:
            # Not yet calibrated - show waiting message
            if self._frame_count % 30 == 0:
                logger.warning(
                    f"Waiting for reference markers: found {ref_markers_found}, "
                    f"need {self.config.min_reference_markers}"
                )
            
            if self.config.display:
                self._draw_calibration_frame(frame, markers, ref_markers_found)
            return
        
        # Detect TurtleBots with YOLO
        turtlebot_detections = []
        if self._yolo_detector:
            yolo_detections = self._yolo_detector.detect(frame)
            
            # Get reference marker pixel positions for filtering
            ref_marker_pixels = []
            for marker_id in markers:
                if marker_id in self.reference_markers:
                    px, py = self.aruco_detector.get_center(markers[marker_id])
                    ref_marker_pixels.append((int(px), int(py)))
            
            # Transform to world coordinates and filter out detections near reference markers
            for det in yolo_detections:
                pixel_x, pixel_y = det["pixel_x"], det["pixel_y"]
                
                # Use 50 pixels as threshold - roughly the size of an ArUco marker
                REF_MARKER_EXCLUSION_RADIUS = 60  # pixels
                near_ref_marker = False
                for ref_px, ref_py in ref_marker_pixels:
                    dist = np.sqrt((pixel_x - ref_px)**2 + (pixel_y - ref_py)**2)
                    if dist < REF_MARKER_EXCLUSION_RADIUS:
                        near_ref_marker = True
                        logger.debug(
                            f"Ignoring YOLO detection at ({pixel_x}, {pixel_y}) - "
                            f"too close to reference marker at ({ref_px}, {ref_py}), "
                            f"dist={dist:.1f}px"
                        )
                        break
                
                if near_ref_marker:
                    continue
                
                world_x, world_y = self._transform_to_world(pixel_x, pixel_y)
                det["x"] = world_x
                det["y"] = world_y
                det["theta"] = 0.0  # YOLO doesn't give orientation
                det["name"] = f"turtlebot"
                turtlebot_detections.append(det)
            
            # Store only valid detections for overlay drawing (excluding ignored ones)
            self._last_yolo_detections = turtlebot_detections
        
        # Update tracker
        tracked = self.tracker.update(
            turtlebot_detections,
            timestamp,
            homography_valid=self._homography_valid
        )
        
        # Publish positions
        if self._sink:
            # Build TurtleBot positions with M{id} naming
            positions = [
                TurtleBotPosition(
                    name=f"M{obj.local_id}",
                    local_id=obj.local_id,
                    x=obj.smoothed_position[0],
                    y=obj.smoothed_position[1],
                    theta=obj.smoothed_position[2],
                    confidence=obj.current_position.confidence if obj.current_position else 0.0,
                    pixel_x=obj.current_position.pixel_x if obj.current_position else 0,
                    pixel_y=obj.current_position.pixel_y if obj.current_position else 0,
                    timestamp=timestamp
                )
                for obj in tracked
            ]
            
            # Build reference marker positions
            ref_marker_positions = []
            for marker_id, marker_cfg in self.reference_markers.items():
                detected = marker_id in markers
                pixel_x, pixel_y = 0, 0
                if detected:
                    pixel_x, pixel_y = self.aruco_detector.get_center(markers[marker_id])
                    pixel_x, pixel_y = int(pixel_x), int(pixel_y)
                
                ref_marker_positions.append(ReferenceMarkerPosition(
                    name=f"R{marker_cfg.r_index}",
                    marker_id=marker_id,
                    world_x=marker_cfg.world_x,
                    world_y=marker_cfg.world_y,
                    pixel_x=pixel_x,
                    pixel_y=pixel_y,
                    detected=detected
                ))
            
            self._sink.publish(
                positions,
                ref_marker_positions,
                timestamp,
                frame_number,
                self._homography_valid
            )
        
        # Update plotter with BOT and REF positions
        if self._plotter and self._homography_valid:
            for obj in tracked:
                x, y, _ = obj.smoothed_position
                self._plotter.update(
                    label="BOT",
                    beacon_id=obj.local_id,
                    x=x,
                    y=y,
                    z=0.0
                )
            for marker_id, marker_cfg in self.reference_markers.items():
                if marker_id in markers:
                    self._plotter.update(
                        label="REF",
                        beacon_id=marker_cfg.r_index,
                        x=marker_cfg.world_x,
                        y=marker_cfg.world_y,
                        z=0.0
                    )
        
        # Call user callback
        if self.on_pose_update and tracked:
            self.on_pose_update(tracked)
        
        # Capture raw frame if enabled
        if self._capture_enabled and self._calibrated:
            self._maybe_capture_frame(frame, timestamp)
        
        # Display
        if self.config.display:
            self._draw_frame(frame, markers, ref_markers_found, tracked)
    
    def _maybe_capture_frame(self, frame: np.ndarray, timestamp: float):
        """
        Capture and save raw frame if enough time has passed.
        
        Args:
            frame: Raw unmodified frame
            timestamp: Current timestamp
        """
        current_time = time.time()
        
        if current_time - self._last_capture_time >= self._capture_interval:
            self._last_capture_time = current_time
            self._capture_count += 1
            
            # Generate filename with timestamp
            filename = f"capture_{self._capture_count:05d}_{int(timestamp*1000)}.jpg"
            filepath = self._capture_dir / filename
            
            # Save raw frame (no overlays)
            cv2.imwrite(str(filepath), frame)
            logger.debug(f"Captured frame: {filepath}")
    
    def _load_yolo_model(self):
        """Load the YOLO model."""
        try:
            self._yolo_detector = YOLODetector(
                self._model_path,
                confidence_threshold=self.config.confidence_threshold
            )
            logger.info("YOLO model loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"YOLO model not found: {e}")
            raise ReferenceMarkerError(
                f"YOLO model not found at {self._model_path}. "
                "Train a model with 'yolo_loc train' or provide --model path."
            )
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def _update_homography(self, markers: Dict[int, np.ndarray]):
        """Update homography matrix from detected markers."""
        image_points = []
        world_points = []
        
        for marker_id, marker_cfg in self.reference_markers.items():
            if marker_id in markers:
                center = self.aruco_detector.get_center(markers[marker_id])
                image_points.append(center)
                world_points.append((marker_cfg.world_x, marker_cfg.world_y))
        
        if len(image_points) >= 4:
            image_pts = np.array(image_points, dtype=np.float32)
            world_pts = np.array(world_points, dtype=np.float32)
            
            self._homography, _ = cv2.findHomography(
                image_pts, world_pts, cv2.RANSAC, 5.0
            )
            self._homography_valid = self._homography is not None
    
    def _transform_to_world(self, pixel_x: float, pixel_y: float) -> tuple:
        """Transform pixel coordinates to world coordinates."""
        if self._homography_valid:
            point = np.array([[pixel_x, pixel_y]], dtype=np.float32).reshape(-1, 1, 2)
            world = cv2.perspectiveTransform(point, self._homography)
            return float(world[0, 0, 0]), float(world[0, 0, 1])
        return pixel_x, pixel_y
    
    def _get_screen_size(self) -> tuple:
        """Get screen size. Returns (width, height)."""
        try:
            # Try using tkinter (works on most systems)
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()  # Hide the window
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            return (width, height)
        except Exception:
            pass
        
        try:
            # Try xrandr on Linux
            import subprocess
            output = subprocess.check_output(['xrandr']).decode('utf-8')
            for line in output.split('\n'):
                if '*' in line:  # Current resolution has asterisk
                    parts = line.split()
                    for part in parts:
                        if 'x' in part and part[0].isdigit():
                            w, h = part.split('x')
                            return (int(w), int(h.split('+')[0]))
        except Exception:
            pass
        
        # Default fallback
        return (1920, 1080)
    
    def _scale_to_fit(self, frame: np.ndarray) -> np.ndarray:
        """Scale frame to fit within max display dimensions while maintaining aspect ratio."""
        h, w = frame.shape[:2]
        
        # Check if scaling is needed
        if w <= self._max_display_width and h <= self._max_display_height:
            return frame
        
        # Calculate scale factor to fit within bounds
        scale_w = self._max_display_width / w
        scale_h = self._max_display_height / h
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize with high quality interpolation
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _draw_calibration_frame(
        self,
        frame: np.ndarray,
        markers: Dict[int, np.ndarray],
        found_refs: List[int]
    ):
        """Draw frame during calibration phase."""
        output = frame.copy()
        
        # Draw detected markers
        if markers:
            corners = list(markers.values())
            ids = np.array([[mid] for mid in markers.keys()])
            cv2.aruco.drawDetectedMarkers(output, corners, ids)
        
        # Draw status
        h, w = output.shape[:2]
        
        # Semi-transparent overlay
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        # Status text
        status = f"CALIBRATING - Detecting reference markers..."
        cv2.putText(output, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        found_str = f"Found: {found_refs}"
        needed_str = f"Need: {list(self.reference_markers.keys())}"
        cv2.putText(output, found_str, (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(output, needed_str, (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Scale to fit screen
        output = self._scale_to_fit(output)
        
        cv2.imshow(self._window_name, output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self._running = False
    
    def _draw_frame(
        self,
        frame: np.ndarray,
        markers: Dict[int, np.ndarray],
        found_refs: List[int],
        tracked: List[TrackedObject]
    ):
        """Draw annotated frame."""
        output = frame.copy()
        
        # Draw reference markers
        if markers:
            corners = list(markers.values())
            ids = np.array([[mid] for mid in markers.keys()])
            cv2.aruco.drawDetectedMarkers(output, corners, ids)
        
        # Draw YOLO detection bounding boxes if overlay enabled
        if self.config.overlay_detections and self._last_yolo_detections:
            for det in self._last_yolo_detections:
                bbox = det.get("bbox")
                if bbox:
                    x, y, w, h = bbox
                    conf = det.get("confidence", 0)
                    # Draw bounding box in cyan
                    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    # Draw confidence label
                    conf_label = f"{conf:.0%}"
                    cv2.putText(output, conf_label, (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Colors for different TurtleBots
        colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
        ]
        
        # Draw tracked TurtleBots
        for obj in tracked:
            pos = obj.current_position
            if pos is None:
                continue
            
            color = colors[obj.local_id % len(colors)]
            px, py = pos.pixel_x, pos.pixel_y
            
            # Draw center point
            cv2.circle(output, (px, py), 10, color, -1)
            cv2.circle(output, (px, py), 12, (255, 255, 255), 2)
            
            # Draw label
            x, y, _ = obj.smoothed_position
            if self._homography_valid:
                label = f"M{obj.local_id}: ({x:.2f}, {y:.2f})m"
            else:
                label = f"M{obj.local_id}: ({px}, {py})px"
            
            label += f" [{pos.confidence:.0%}]"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(output, (px + 15, py - th - 5), (px + 20 + tw, py + 5), (0, 0, 0), -1)
            cv2.putText(output, label, (px + 18, py),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw status bar
        h, w = output.shape[:2]
        
        status = f"Refs: {len(found_refs)}/{len(self.reference_markers)} | "
        status += f"Homography: {'OK' if self._homography_valid else 'NO'} | "
        status += f"TurtleBots: {len(tracked)}"
        
        if self._broadcaster:
            status += f" | Clients: {self._broadcaster.client_count}"
        
        cv2.rectangle(output, (0, h - 35), (w, h), (0, 0, 0), -1)
        cv2.putText(output, status, (10, h - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Scale to fit screen
        output = self._scale_to_fit(output)
        
        cv2.imshow(self._window_name, output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self._running = False
        elif key == ord('r'):
            # Reset calibration
            self._calibrated = False
            self._homography_valid = False
            self.tracker.reset()
            logger.info("Calibration reset")
    
    def _check_stream_timeout(self):
        """Check if stream has timed out."""
        if self._last_frame_time == 0:
            return
        
        elapsed = time.time() - self._last_frame_time
        
        if elapsed > self.config.stream_timeout and not self._paused:
            logger.warning(
                f"No frames for {elapsed:.1f}s - pausing broadcast/logging"
            )
            self._paused = True
            if self._sink:
                self._sink.pause()
    
    def start(self, auto_reconnect: bool = True):
        """
        Start the localizer.
        
        Args:
            auto_reconnect: Whether to auto-reconnect on connection loss
        """
        self._running = True
        
        # Start broadcaster if configured
        if self._broadcaster:
            if not self._broadcaster.start():
                logger.error("Failed to start broadcaster")
                return
        
        logger.info("Starting TurtleBot Localizer...")
        logger.info("Controls: q=quit, r=reset calibration")
        
        # Start timeout checker thread
        import threading
        def timeout_checker():
            while self._running:
                self._check_stream_timeout()
                time.sleep(0.5)
        
        timeout_thread = threading.Thread(target=timeout_checker, daemon=True)
        timeout_thread.start()
        
        try:
            self.listener.start(auto_reconnect=auto_reconnect)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the localizer."""
        self._running = False
        self.listener.stop()
        
        if self.config.display:
            cv2.destroyAllWindows()
        
        if self._sink:
            self._sink.close()
        
        if self._plotter:
            self._plotter.close()
        
        # Print statistics
        logger.info(f"Localizer stopped. Processed {self._frame_count} frames")
        logger.info(f"Tracker stats: {self.tracker.stats}")


def load_config(config_path: str) -> LocalizerConfig:
    """
    Load localizer configuration from JSON file.
    
    Expected format:
    {
        "host": "100.99.98.1",
        "port": 5000,
        "reference_markers": [
            {"id": 0, "x": 0.0, "y": 0.0},
            {"id": 1, "x": 2.0, "y": 0.0},
            ...
        ],
        "model_path": "path/to/model.pt",
        "confidence_threshold": 0.5,
        "broadcast_port": 5555,
        "stream_timeout": 3.0
    }
    """
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    ref_markers = None
    if "reference_markers" in data:
        ref_markers = [
            MarkerConfig(
                marker_id=m["id"],
                world_x=m["x"],
                world_y=m["y"]
            )
            for m in data["reference_markers"]
        ]
    
    return LocalizerConfig(
        host=data.get("host", "100.99.98.1"),
        port=data.get("port", 5000),
        reference_markers=ref_markers,
        model_path=data.get("model_path"),
        confidence_threshold=data.get("confidence_threshold", 0.5),
        broadcast_port=data.get("broadcast_port", 5555),
        broadcast_rate_hz=data.get("broadcast_rate_hz", 20.0),
        csv_rate_hz=data.get("csv_rate_hz", 10.0),
        stream_timeout=data.get("stream_timeout", 3.0),
        display=data.get("display", True)
    )


def main():
    parser = argparse.ArgumentParser(
        description="TurtleBot Localizer - Real-time localization from camera stream",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Basic usage (display only):
        python localizer.py --host 100.99.98.1
    
    With broadcasting:
        python localizer.py --host 100.99.98.1 --broadcast
    
    With CSV logging:
        python localizer.py --host 100.99.98.1 --csv positions.csv
    
    Full setup:
        python localizer.py --host 100.99.98.1 --broadcast --csv positions.csv
    
    Custom model:
        python localizer.py --host 100.99.98.1 --model path/to/best.pt
    
    Load config from file:
        python localizer.py --config localizer_config.json

Controls (when display is enabled):
    q - Quit
    r - Reset calibration
        """
    )
    
    parser.add_argument("--host", default="100.99.98.1",
                       help="Streamer host address (default: 100.99.98.1)")
    parser.add_argument("--port", type=int, default=5000,
                       help="Streamer port (default: 5000)")
    parser.add_argument("--config", type=str,
                       help="Path to JSON configuration file")
    parser.add_argument("--model", "-m", type=str,
                       help="Path to YOLO model (default: files/models/yolo/.../best.pt)")
    parser.add_argument("--confidence", "-c", type=float, default=0.85,
                       help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--broadcast", "-b", action="store_true",
                       help="Enable position broadcasting")
    parser.add_argument("--broadcast-port", type=int, default=5555,
                       help="Broadcast port (default: 5555)")
    parser.add_argument("--csv", type=str,
                       help="Path to save CSV positions")
    parser.add_argument("--no-display", action="store_true",
                       help="Disable display window")
    parser.add_argument("--plot", "-p", action="store_true",
                       help="Enable position plotter (matplotlib)")
    parser.add_argument("--no-reconnect", action="store_true",
                       help="Disable auto-reconnect")
    parser.add_argument("--overlay", "-o", action="store_true",
                       help="Overlay YOLO detection bounding boxes on display")
    parser.add_argument("--timeout", type=float, default=3.0,
                       help="Stream timeout in seconds (default: 3.0)")
    parser.add_argument("--log-level", default="info",
                       choices=["debug", "info", "warning", "error"],
                       help="Logging level (default: info)")
    parser.add_argument("--capture-images", "-t", action="store_true",
                       help="Enable periodic capture of raw frames")
    parser.add_argument("--capture-interval", type=float, default=5.0,
                       help="Interval between captures in seconds (default: 5.0)")
    parser.add_argument("--capture-dir", type=str,
                       help=f"Directory to save captured images (default: {_DEFAULT_CAPTURE_DIR})")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        experiment_name="localizer",
        level=args.log_level,
        log_to_file=True,
        log_to_console=True
    )
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load or create configuration
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = LocalizerConfig(
            host=args.host,
            port=args.port,
            model_path=args.model,
            confidence_threshold=args.confidence,
            broadcast_port=args.broadcast_port,
            stream_timeout=args.timeout,
            display=not args.no_display,
            overlay_detections=args.overlay,
            capture_images=args.capture_images,
            capture_interval=args.capture_interval,
            capture_dir=args.capture_dir
        )
    
    # Create and start localizer
    localizer = TurtleBotLocalizer(
        config=config,
        csv_path=args.csv,
        broadcast=args.broadcast,
        plot=args.plot
    )
    
    localizer.start(auto_reconnect=not args.no_reconnect)


if __name__ == "__main__":
    main()
