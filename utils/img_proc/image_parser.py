#!/usr/bin/env python3
"""
Image Parser for TurtleBot Localization

Uses ArUco markers for reference positions (world coordinate system)
and YOLO object detection for tracking TurtleBots.

Architecture:
    - Reference markers: Fixed ArUco markers at known world positions
    - TurtleBot detection: YOLO object detection (trained model)
    - Homography-based transformation from image to world coordinates
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# YOLO import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed. Install with: pip install ultralytics")


@dataclass
class MarkerConfig:
    """Configuration for a reference ArUco marker."""
    marker_id: int
    world_position: Tuple[float, float]  # (x, y) in world coordinates (meters)


@dataclass
class TurtleBotConfig:
    """Configuration for a TurtleBot."""
    name: str  # e.g., "tb3_0", "tb3_1", "tb3_2"
    class_id: int = 0  # YOLO class ID
    min_confidence: float = 0.5  # Minimum detection confidence


@dataclass
class TurtleBotPose:
    """Pose of a detected TurtleBot."""
    name: str
    x: float  # meters (or pixels if no homography)
    y: float  # meters (or pixels if no homography)
    theta: float  # radians (orientation, 0 if not determinable)
    confidence: float  # detection confidence (0-1)
    pixel_position: Tuple[int, int]  # (u, v) in image
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) bounding box
    class_id: int = 0  # YOLO class ID
    timestamp: float = 0.0


@dataclass
class LocalizationResult:
    """Result of localization for a single frame."""
    timestamp: float
    turtlebots: List[TurtleBotPose]
    reference_markers_detected: int
    homography_valid: bool
    frame_number: int = 0


class ArucoDetector:
    """Handles ArUco marker detection for reference markers."""
    
    def __init__(self, dictionary_type: int = cv2.aruco.DICT_4X4_50):
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        
    def detect(self, frame: np.ndarray) -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]:
        """Detect ArUco markers in frame."""
        corners, ids, rejected = self.detector.detectMarkers(frame)
        if ids is None:
            return [], [], rejected
        return ids.flatten().tolist(), corners, rejected
    
    def get_marker_center(self, corners: np.ndarray) -> Tuple[float, float]:
        """Get the center point of a marker from its corners."""
        center = corners.mean(axis=1).flatten()
        return (center[0], center[1])


class ObjectDetector:
    """
    YOLO-based object detector for TurtleBot tracking.
    
    Uses ultralytics YOLO for robust detection across varying
    lighting conditions and viewpoints.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto"
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model (.pt file). If None, uses yolov8n.
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run on ("auto", "cpu", "cuda", "mps")
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        if model_path is None:
            logger.info("No model path provided, using pretrained yolov8n")
            self.model = YOLO("yolov8n.pt")
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            logger.info(f"Loading YOLO model from {model_path}")
            self.model = YOLO(str(model_path))
        
        # Set device
        if device != "auto":
            self.model.to(device)
        
        # Get class names
        self.class_names = self.model.names
        logger.info(f"Model loaded with classes: {self.class_names}")
    
    def detect(
        self, 
        frame: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Detect objects in frame.
        
        Args:
            frame: BGR image
            classes: List of class IDs to detect (None = all)
            
        Returns:
            List of detections with keys: center, bbox, confidence, class_id, class_name
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=classes,
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
                
                w = x2 - x1
                h = y2 - y1
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                conf = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.class_names.get(class_id, str(class_id))
                
                detections.append({
                    "center": (cx, cy),
                    "bbox": (x1, y1, w, h),
                    "confidence": conf,
                    "class_id": class_id,
                    "class_name": class_name,
                    "area": w * h
                })
        
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections


class ImageParser:
    """
    Parses camera images to localize TurtleBots.
    
    Uses ArUco markers for reference positions and YOLO
    object detection for TurtleBot tracking.
    """
    
    def __init__(
        self,
        reference_markers: List[MarkerConfig],
        turtlebot_configs: List[TurtleBotConfig],
        aruco_dict: int = cv2.aruco.DICT_4X4_50,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the image parser.
        
        Args:
            reference_markers: List of reference ArUco marker configurations
            turtlebot_configs: List of TurtleBot configurations
            aruco_dict: ArUco dictionary type
            model_path: Path to YOLO model
            confidence_threshold: Minimum confidence for detections
        """
        self.reference_markers = {m.marker_id: m for m in reference_markers}
        self.turtlebot_configs = turtlebot_configs
        
        self.aruco_detector = ArucoDetector(aruco_dict)
        self.object_detector = ObjectDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        
        # Cached homography matrix
        self._homography: Optional[np.ndarray] = None
        self._homography_valid = False
        
        # Statistics
        self._frames_processed = 0
        self._successful_localizations = 0
        
        logger.info(f"ImageParser initialized with {len(reference_markers)} reference markers, "
                   f"{len(turtlebot_configs)} TurtleBots")
    
    def process_frame(
        self, 
        frame: np.ndarray, 
        timestamp: float = 0.0,
        frame_number: int = 0
    ) -> LocalizationResult:
        """
        Process a frame and localize TurtleBots.
        
        Args:
            frame: BGR image from camera
            timestamp: Frame timestamp
            frame_number: Frame number
            
        Returns:
            LocalizationResult with detected TurtleBot poses
        """
        self._frames_processed += 1
        
        # Step 1: Detect ArUco reference markers and update homography
        marker_ids, corners, _ = self.aruco_detector.detect(frame)
        marker_corners = {mid: corners[i] for i, mid in enumerate(marker_ids)}
        ref_detected = self._update_homography(marker_corners)
        
        # Step 2: Detect TurtleBots with YOLO
        turtlebot_poses = self._detect_turtlebots(frame, timestamp)
        
        if turtlebot_poses:
            self._successful_localizations += 1
        
        return LocalizationResult(
            timestamp=timestamp,
            turtlebots=turtlebot_poses,
            reference_markers_detected=ref_detected,
            homography_valid=self._homography_valid,
            frame_number=frame_number
        )
    
    def _detect_turtlebots(
        self, 
        frame: np.ndarray, 
        timestamp: float
    ) -> List[TurtleBotPose]:
        """Detect TurtleBots using YOLO object detection."""
        # Get class IDs to detect
        class_ids = list(set(tb.class_id for tb in self.turtlebot_configs))
        
        # Detect objects
        detections = self.object_detector.detect(frame, classes=class_ids)
        
        poses = []
        assigned_configs = set()
        
        for detection in detections:
            # Match detection to TurtleBot config by class_id
            for tb_config in self.turtlebot_configs:
                if tb_config.name in assigned_configs:
                    continue
                    
                if tb_config.class_id == detection["class_id"]:
                    if detection["confidence"] >= tb_config.min_confidence:
                        pose = self._create_pose(detection, tb_config, timestamp)
                        poses.append(pose)
                        assigned_configs.add(tb_config.name)
                        break
        
        return poses
    
    def _update_homography(self, marker_corners: Dict[int, np.ndarray]) -> int:
        """Update homography matrix using detected reference markers."""
        image_points = []
        world_points = []
        
        for marker_id, ref_config in self.reference_markers.items():
            if marker_id in marker_corners:
                center = self.aruco_detector.get_marker_center(marker_corners[marker_id])
                image_points.append(center)
                world_points.append(ref_config.world_position)
        
        ref_detected = len(image_points)
        
        if ref_detected >= 4:
            image_pts = np.array(image_points, dtype=np.float32)
            world_pts = np.array(world_points, dtype=np.float32)
            
            self._homography, _ = cv2.findHomography(image_pts, world_pts, cv2.RANSAC, 5.0)
            self._homography_valid = self._homography is not None
        
        return ref_detected
    
    def _transform_to_world(
        self, 
        pixel_x: float, 
        pixel_y: float
    ) -> Tuple[float, float, float]:
        """Transform pixel coordinates to world coordinates."""
        if self._homography_valid:
            point = np.array([[pixel_x, pixel_y]], dtype=np.float32).reshape(-1, 1, 2)
            world_point = cv2.perspectiveTransform(point, self._homography)
            return world_point[0, 0, 0], world_point[0, 0, 1], 1.0
        else:
            return pixel_x, pixel_y, 0.5
    
    def _create_pose(
        self, 
        detection: Dict, 
        config: TurtleBotConfig, 
        timestamp: float
    ) -> TurtleBotPose:
        """Create TurtleBotPose from YOLO detection."""
        cx, cy = detection["center"]
        x, y, confidence = self._transform_to_world(cx, cy)
        
        return TurtleBotPose(
            name=config.name,
            x=x,
            y=y,
            theta=0.0,
            confidence=min(confidence, detection["confidence"]),
            pixel_position=(int(cx), int(cy)),
            bbox=detection["bbox"],
            class_id=detection["class_id"],
            timestamp=timestamp
        )
    
    def draw_detections(
        self, 
        frame: np.ndarray, 
        result: LocalizationResult,
        draw_reference: bool = True,
        draw_bbox: bool = True
    ) -> np.ndarray:
        """
        Draw detection results on frame.
        
        Args:
            frame: Original BGR frame
            result: Localization result
            draw_reference: Whether to draw reference markers
            draw_bbox: Whether to draw bounding boxes
            
        Returns:
            Annotated frame
        """
        output = frame.copy()
        
        # Draw ArUco markers
        if draw_reference:
            marker_ids, corners, _ = self.aruco_detector.detect(frame)
            if marker_ids:
                cv2.aruco.drawDetectedMarkers(output, corners, np.array(marker_ids))
        
        # Colors for different TurtleBots
        colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green  
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
        ]
        
        # Draw TurtleBot detections
        for i, tb in enumerate(result.turtlebots):
            px, py = tb.pixel_position
            color_bgr = colors[i % len(colors)]
            
            # Draw bounding box
            if draw_bbox and tb.bbox:
                bx, by, bw, bh = tb.bbox
                cv2.rectangle(output, (bx, by), (bx + bw, by + bh), color_bgr, 2)
            
            # Draw center point
            cv2.circle(output, (px, py), 8, color_bgr, -1)
            cv2.circle(output, (px, py), 10, (255, 255, 255), 2)
            
            # Draw label
            if self._homography_valid:
                label = f"{tb.name}: ({tb.x:.2f}, {tb.y:.2f})m"
            else:
                label = f"{tb.name}: ({px}, {py})px"
            
            label += f" [{tb.confidence:.0%}]"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(output, (px + 15, py - th - 5), (px + 20 + tw, py + 5), (0, 0, 0), -1)
            cv2.putText(output, label, (px + 18, py),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
        
        # Draw status bar
        status = f"YOLO | Ref: {result.reference_markers_detected} | "
        status += f"Homography: {'OK' if result.homography_valid else 'NO'} | "
        status += f"TBs: {len(result.turtlebots)}"
        
        cv2.rectangle(output, (0, output.shape[0] - 30), (450, output.shape[0]), (0, 0, 0), -1)
        cv2.putText(output, status, (10, output.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output
    
    @property
    def stats(self) -> Dict:
        """Get processing statistics."""
        return {
            "frames_processed": self._frames_processed,
            "successful_localizations": self._successful_localizations,
            "success_rate": (self._successful_localizations / self._frames_processed * 100 
                          if self._frames_processed > 0 else 0)
        }


def create_default_config() -> Tuple[List[MarkerConfig], List[TurtleBotConfig]]:
    """
    Create default marker configuration.
    
    Default setup:
    - 4 ArUco reference markers at corners of a 2m x 2m area
    - 3 TurtleBots with class_id 0 (turtlebot)
    """
    reference_markers = [
        MarkerConfig(marker_id=0, world_position=(0.0, 0.0)),
        MarkerConfig(marker_id=1, world_position=(2.0, 0.0)),
        MarkerConfig(marker_id=2, world_position=(2.0, 2.0)),
        MarkerConfig(marker_id=3, world_position=(0.0, 2.0)),
    ]
    
    turtlebot_configs = [
        TurtleBotConfig(name="tb3_0", class_id=0),
        TurtleBotConfig(name="tb3_1", class_id=0),
        TurtleBotConfig(name="tb3_2", class_id=0),
    ]
    
    return reference_markers, turtlebot_configs


def create_parser(
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.5
) -> ImageParser:
    """
    Create ImageParser with default configuration.
    
    Args:
        model_path: Path to YOLO model trained on TurtleBots
        confidence_threshold: Minimum confidence for detections
    """
    ref_markers, tb_configs = create_default_config()
    return ImageParser(
        ref_markers, 
        tb_configs,
        model_path=model_path,
        confidence_threshold=confidence_threshold
    )
