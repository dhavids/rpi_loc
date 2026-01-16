"""
TurtleBot Position Tracker

Tracks TurtleBot positions across frames with consistency checking
to maintain stable object IDs even when detections are noisy.
"""

import time
import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackedPosition:
    """A single tracked position sample."""
    timestamp: float
    x: float
    y: float
    theta: float
    confidence: float
    pixel_x: int
    pixel_y: int


@dataclass
class TrackedObject:
    """
    A tracked TurtleBot with position history.
    
    Maintains a history of positions for smoothing and consistency checking.
    """
    local_id: int
    name: str
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    last_seen: float = 0.0
    
    # Smoothed position (EMA)
    ema_x: Optional[float] = None
    ema_y: Optional[float] = None
    ema_theta: Optional[float] = None
    
    # Detection statistics
    detection_count: int = 0
    consecutive_misses: int = 0
    
    @property
    def current_position(self) -> Optional[TrackedPosition]:
        """Get the most recent position."""
        return self.history[-1] if self.history else None
    
    @property
    def smoothed_position(self) -> Tuple[float, float, float]:
        """Get smoothed position (x, y, theta)."""
        if self.ema_x is not None:
            return (self.ema_x, self.ema_y, self.ema_theta)
        elif self.history:
            pos = self.history[-1]
            return (pos.x, pos.y, pos.theta)
        return (0.0, 0.0, 0.0)


class PositionTracker:
    """
    Tracks TurtleBot positions across frames.
    
    Provides:
    - Consistent local IDs for detected TurtleBots
    - Position smoothing using EMA
    - Association between frames using distance-based matching
    - Timeout handling for lost objects
    - Re-identification of recently lost objects
    """
    
    # Configuration
    MAX_ASSOCIATION_DISTANCE = 0.5  # meters (or pixels if no homography)
    MAX_PIXEL_DISTANCE = 100  # pixels for association
    EMA_ALPHA = 0.5  # Smoothing factor (higher = more responsive)
    LOST_TIMEOUT = 2.0  # seconds before object is considered lost
    MIN_CONFIDENCE = 0.3  # Minimum confidence to track
    
    # Re-identification configuration
    REIDENTIFY_TIMEOUT = 30.0  # seconds to keep lost objects for re-identification
    REIDENTIFY_DISTANCE = 1.0  # meters - max distance to re-identify
    REIDENTIFY_PIXEL_DISTANCE = 200  # pixels - max distance to re-identify
    
    def __init__(
        self,
        use_smoothing: bool = True,
        max_objects: int = 10,
        use_pixel_association: bool = True
    ):
        """
        Initialize the tracker.
        
        Args:
            use_smoothing: Whether to apply EMA smoothing
            max_objects: Maximum number of objects to track
            use_pixel_association: Use pixel distance for association (useful without homography)
        """
        self.use_smoothing = use_smoothing
        self.max_objects = max_objects
        self.use_pixel_association = use_pixel_association
        
        self._objects: Dict[int, TrackedObject] = {}
        self._lost_objects: Dict[int, TrackedObject] = {}  # Recently lost, for re-identification
        self._next_id = 0  # 0-indexed: M0, M1, M2, ...
        self._frame_count = 0
    
    def update(
        self,
        detections: List[Dict],
        timestamp: float,
        homography_valid: bool = False
    ) -> List[TrackedObject]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with keys:
                - name: str (optional, default "turtlebot")
                - x: float (world x)
                - y: float (world y)
                - theta: float (orientation)
                - confidence: float
                - pixel_x: int
                - pixel_y: int
            timestamp: Frame timestamp
            homography_valid: Whether world coordinates are valid
            
        Returns:
            List of currently tracked objects
        """
        self._frame_count += 1
        
        # Filter low-confidence detections
        valid_detections = [
            d for d in detections 
            if d.get("confidence", 0) >= self.MIN_CONFIDENCE
        ]
        
        # Associate detections with existing tracks
        associations = self._associate_detections(
            valid_detections, 
            timestamp,
            homography_valid
        )
        
        # Update matched tracks
        matched_ids = set()
        for detection, obj_id in associations:
            self._update_object(obj_id, detection, timestamp)
            matched_ids.add(obj_id)
        
        # Create new tracks for unmatched detections
        unmatched_detections = [
            d for d in valid_detections 
            if all(d is not assoc[0] for assoc in associations)
        ]
        
        for detection in unmatched_detections:
            if len(self._objects) < self.max_objects:
                self._create_object(detection, timestamp, homography_valid)
        
        # Update miss counts and remove lost objects
        self._update_lost_objects(timestamp, matched_ids)
        
        # Return active objects
        return list(self._objects.values())
    
    def _associate_detections(
        self,
        detections: List[Dict],
        timestamp: float,
        homography_valid: bool
    ) -> List[Tuple[Dict, int]]:
        """
        Associate detections with existing tracked objects.
        
        Uses Hungarian algorithm approximation via greedy matching.
        """
        if not detections or not self._objects:
            return []
        
        associations = []
        used_detections = set()
        used_objects = set()
        
        # Build cost matrix (distance between each detection and object)
        costs = []
        for det_idx, detection in enumerate(detections):
            for obj_id, obj in self._objects.items():
                if obj.current_position is None:
                    continue
                
                # Calculate distance
                if homography_valid and not self.use_pixel_association:
                    # World coordinate distance
                    dist = self._world_distance(
                        detection, obj.current_position
                    )
                    max_dist = self.MAX_ASSOCIATION_DISTANCE
                else:
                    # Pixel distance
                    dist = self._pixel_distance(
                        detection, obj.current_position
                    )
                    max_dist = self.MAX_PIXEL_DISTANCE
                
                if dist < max_dist:
                    costs.append((dist, det_idx, obj_id))
        
        # Greedy matching (sort by cost, assign lowest first)
        costs.sort(key=lambda x: x[0])
        
        for cost, det_idx, obj_id in costs:
            if det_idx in used_detections or obj_id in used_objects:
                continue
            
            associations.append((detections[det_idx], obj_id))
            used_detections.add(det_idx)
            used_objects.add(obj_id)
        
        return associations
    
    def _world_distance(self, detection: Dict, position: TrackedPosition) -> float:
        """Calculate distance in world coordinates."""
        dx = detection.get("x", 0) - position.x
        dy = detection.get("y", 0) - position.y
        return math.sqrt(dx * dx + dy * dy)
    
    def _pixel_distance(self, detection: Dict, position: TrackedPosition) -> float:
        """Calculate distance in pixel coordinates."""
        dx = detection.get("pixel_x", 0) - position.pixel_x
        dy = detection.get("pixel_y", 0) - position.pixel_y
        return math.sqrt(dx * dx + dy * dy)
    
    def _update_object(self, obj_id: int, detection: Dict, timestamp: float):
        """Update an existing tracked object with new detection."""
        obj = self._objects[obj_id]
        
        x = detection.get("x", 0.0)
        y = detection.get("y", 0.0)
        theta = detection.get("theta", 0.0)
        confidence = detection.get("confidence", 0.5)
        pixel_x = detection.get("pixel_x", 0)
        pixel_y = detection.get("pixel_y", 0)
        
        # Update name if provided and different
        if "name" in detection and detection["name"]:
            obj.name = detection["name"]
        
        # Apply EMA smoothing
        if self.use_smoothing:
            if obj.ema_x is None:
                obj.ema_x = x
                obj.ema_y = y
                obj.ema_theta = theta
            else:
                alpha = self.EMA_ALPHA
                obj.ema_x = alpha * x + (1 - alpha) * obj.ema_x
                obj.ema_y = alpha * y + (1 - alpha) * obj.ema_y
                
                # Handle angle wrapping for theta
                theta_diff = theta - obj.ema_theta
                if theta_diff > math.pi:
                    theta_diff -= 2 * math.pi
                elif theta_diff < -math.pi:
                    theta_diff += 2 * math.pi
                obj.ema_theta = obj.ema_theta + alpha * theta_diff
        
        # Create position sample
        pos = TrackedPosition(
            timestamp=timestamp,
            x=obj.ema_x if self.use_smoothing and obj.ema_x else x,
            y=obj.ema_y if self.use_smoothing and obj.ema_y else y,
            theta=obj.ema_theta if self.use_smoothing and obj.ema_theta else theta,
            confidence=confidence,
            pixel_x=pixel_x,
            pixel_y=pixel_y
        )
        
        obj.history.append(pos)
        obj.last_seen = timestamp
        obj.detection_count += 1
        obj.consecutive_misses = 0
    
    def _create_object(self, detection: Dict, timestamp: float, homography_valid: bool = False):
        """
        Create a new tracked object, or re-identify a recently lost one.
        
        First checks if the detection is near a recently lost object's last position.
        If so, re-uses that object's ID for continuity.
        """
        # Check if this detection matches a recently lost object
        reidentified_obj = self._try_reidentify(detection, timestamp, homography_valid)
        
        if reidentified_obj is not None:
            obj_id = reidentified_obj.local_id
            obj = reidentified_obj
            
            # Remove from lost objects
            del self._lost_objects[obj_id]
            
            # Reset consecutive misses
            obj.consecutive_misses = 0
            
            logger.info(
                f"Re-identified TurtleBot {obj.name} (ID {obj_id}) - "
                f"was lost for {timestamp - obj.last_seen:.1f}s"
            )
        else:
            # Create brand new object
            obj_id = self._next_id
            self._next_id += 1
            
            name = detection.get("name", f"turtlebot_{obj_id}")
            
            obj = TrackedObject(
                local_id=obj_id,
                name=name
            )
            
            logger.info(f"New TurtleBot tracked: {name} (ID {obj_id})")
        
        # Initialize/update with detection
        x = detection.get("x", 0.0)
        y = detection.get("y", 0.0)
        theta = detection.get("theta", 0.0)
        
        if self.use_smoothing:
            obj.ema_x = x
            obj.ema_y = y
            obj.ema_theta = theta
        
        pos = TrackedPosition(
            timestamp=timestamp,
            x=x,
            y=y,
            theta=theta,
            confidence=detection.get("confidence", 0.5),
            pixel_x=detection.get("pixel_x", 0),
            pixel_y=detection.get("pixel_y", 0)
        )
        
        obj.history.append(pos)
        obj.last_seen = timestamp
        obj.detection_count += 1
        
        self._objects[obj_id] = obj
    
    def _try_reidentify(
        self, 
        detection: Dict, 
        timestamp: float,
        homography_valid: bool
    ) -> Optional[TrackedObject]:
        """
        Try to re-identify a recently lost object based on position proximity.
        
        Returns the matched TrackedObject if found, None otherwise.
        """
        if not self._lost_objects:
            return None
        
        best_match = None
        best_distance = float('inf')
        
        det_x = detection.get("x", 0.0)
        det_y = detection.get("y", 0.0)
        det_px = detection.get("pixel_x", 0)
        det_py = detection.get("pixel_y", 0)
        
        for obj_id, obj in self._lost_objects.items():
            # Check if still within re-identification window
            time_since_lost = timestamp - obj.last_seen
            if time_since_lost > self.REIDENTIFY_TIMEOUT:
                continue
            
            # Get last known position
            last_pos = obj.current_position
            if last_pos is None:
                continue
            
            # Calculate distance
            if homography_valid and not self.use_pixel_association:
                # World coordinate distance
                dx = det_x - last_pos.x
                dy = det_y - last_pos.y
                dist = math.sqrt(dx * dx + dy * dy)
                max_dist = self.REIDENTIFY_DISTANCE
            else:
                # Pixel distance
                dx = det_px - last_pos.pixel_x
                dy = det_py - last_pos.pixel_y
                dist = math.sqrt(dx * dx + dy * dy)
                max_dist = self.REIDENTIFY_PIXEL_DISTANCE
            
            if dist < max_dist and dist < best_distance:
                best_distance = dist
                best_match = obj
        
        return best_match

    def _update_lost_objects(self, timestamp: float, matched_ids: set):
        """Update miss counts and handle lost objects."""
        lost_ids = []
        
        for obj_id, obj in self._objects.items():
            if obj_id not in matched_ids:
                obj.consecutive_misses += 1
                
                # Check if lost
                time_since_seen = timestamp - obj.last_seen
                if time_since_seen > self.LOST_TIMEOUT:
                    lost_ids.append(obj_id)
                    logger.info(
                        f"Lost TurtleBot {obj.name} (ID {obj_id}) - "
                        f"not seen for {time_since_seen:.1f}s"
                    )
        
        # Move lost objects to cache for potential re-identification
        for obj_id in lost_ids:
            self._lost_objects[obj_id] = self._objects[obj_id]
            del self._objects[obj_id]
        
        # Clean up expired objects from lost cache
        expired_ids = []
        for obj_id, obj in self._lost_objects.items():
            time_since_lost = timestamp - obj.last_seen
            if time_since_lost > self.REIDENTIFY_TIMEOUT:
                expired_ids.append(obj_id)
        
        for obj_id in expired_ids:
            logger.debug(
                f"Removing expired lost object {self._lost_objects[obj_id].name} "
                f"(ID {obj_id}) from re-identification cache"
            )
            del self._lost_objects[obj_id]
    
    def get_tracked_objects(self) -> List[TrackedObject]:
        """Get all currently tracked objects."""
        return list(self._objects.values())
    
    def get_object_by_id(self, local_id: int) -> Optional[TrackedObject]:
        """Get a specific tracked object by ID."""
        return self._objects.get(local_id)
    
    def reset(self):
        """Reset all tracking state."""
        self._objects.clear()
        self._lost_objects.clear()
        self._next_id = 0
        self._frame_count = 0
        logger.info("Tracker reset")
    
    @property
    def object_count(self) -> int:
        """Get number of currently tracked objects."""
        return len(self._objects)
    
    @property
    def stats(self) -> Dict:
        """Get tracking statistics."""
        return {
            "frame_count": self._frame_count,
            "tracked_objects": len(self._objects),
            "lost_objects_in_cache": len(self._lost_objects),
            "next_id": self._next_id,
            "objects": {
                obj.local_id: {
                    "name": obj.name,
                    "detection_count": obj.detection_count,
                    "last_seen": obj.last_seen
                }
                for obj in self._objects.values()
            }
        }
