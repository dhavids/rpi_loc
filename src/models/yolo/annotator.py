#!/usr/bin/env python3
"""
Image Annotator for YOLO Training Data

Provides tools for manually annotating images with bounding boxes
for TurtleBot detection training.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not installed. Install with: pip install opencv-python")

from .dataset import BoundingBox, Annotation


@dataclass
class AnnotationState:
    """State for annotation session."""
    current_box: Optional[List[int]] = None  # [x1, y1, x2, y2]
    drawing: bool = False
    boxes: List[BoundingBox] = None
    class_id: int = 0
    
    def __post_init__(self):
        if self.boxes is None:
            self.boxes = []


class ImageAnnotator:
    """
    Interactive image annotator for creating YOLO training data.
    
    Controls:
        - Left click + drag: Draw bounding box
        - Right click: Delete last box
        - 'n' or Space: Next image
        - 'p' or Backspace: Previous image
        - 's': Save annotations
        - 'c': Change class (cycles through classes)
        - 'u': Undo last box
        - 'r': Reset all boxes for current image
        - 'q' or Esc: Quit
    
    Example:
        >>> annotator = ImageAnnotator("images/", "labels/", ["turtlebot"])
        >>> annotator.run()
    """
    
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        class_names: List[str] = None,
        window_name: str = "Image Annotator",
        max_window_size: Tuple[int, int] = (1280, 720)
    ):
        """
        Initialize annotator.
        
        Args:
            images_dir: Directory containing images to annotate
            labels_dir: Directory to save YOLO label files
            class_names: List of class names
            window_name: OpenCV window name
            max_window_size: Maximum (width, height) for display window
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required. Install with: pip install opencv-python")
        
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names = class_names or ["turtlebot"]
        self.window_name = window_name
        self.max_window_size = max_window_size
        
        # Ensure labels directory exists
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of images
        self.image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ])
        
        if not self.image_files:
            raise ValueError(f"No images found in {images_dir}")
        
        logger.info(f"Found {len(self.image_files)} images to annotate")
        
        # State
        self.current_idx = 0
        self.state = AnnotationState()
        self.current_image = None
        self.display_image = None
        self.img_height = 0
        self.img_width = 0
        self.scale = 1.0  # Scale factor for display
        
        # Colors for different classes
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
    
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse events for drawing boxes."""
        # Convert display coordinates to original image coordinates
        orig_x = int(x / self.scale)
        orig_y = int(y / self.scale)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.state.drawing = True
            self.state.current_box = [orig_x, orig_y, orig_x, orig_y]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.state.drawing and self.state.current_box:
                self.state.current_box[2] = orig_x
                self.state.current_box[3] = orig_y
                self._draw()
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.state.drawing and self.state.current_box:
                self.state.drawing = False
                x1, y1, x2, y2 = self.state.current_box
                
                # Ensure x1 < x2 and y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Only add if box has reasonable size
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    box = BoundingBox.from_xyxy(
                        self.state.class_id,
                        x1, y1, x2, y2,
                        self.img_width, self.img_height
                    )
                    self.state.boxes.append(box)
                
                self.state.current_box = None
                self._draw()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Delete last box
            if self.state.boxes:
                self.state.boxes.pop()
                self._draw()
    
    def _draw(self) -> None:
        """Draw current state on image."""
        self.display_image = self.current_image.copy()
        
        # Draw existing boxes
        for box in self.state.boxes:
            x1, y1, x2, y2 = box.to_xyxy(self.img_width, self.img_height)
            color = self.colors[box.class_id % len(self.colors)]
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw class label
            label = self.class_names[box.class_id]
            cv2.putText(self.display_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw current box being drawn
        if self.state.current_box:
            x1, y1, x2, y2 = self.state.current_box
            color = self.colors[self.state.class_id % len(self.colors)]
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw info bar
        self._draw_info_bar()
        
        # Scale for display
        if self.scale != 1.0:
            display_w = int(self.img_width * self.scale)
            display_h = int(self.img_height * self.scale)
            self.display_image = cv2.resize(self.display_image, (display_w, display_h))
        
        cv2.imshow(self.window_name, self.display_image)
    
    def _draw_info_bar(self) -> None:
        """Draw information bar at top of image."""
        bar_height = 40
        cv2.rectangle(self.display_image, (0, 0), (self.img_width, bar_height), (0, 0, 0), -1)
        
        # Current image info
        img_name = self.image_files[self.current_idx].name
        progress = f"[{self.current_idx + 1}/{len(self.image_files)}]"
        class_name = self.class_names[self.state.class_id]
        n_boxes = len(self.state.boxes)
        
        info = f"{progress} {img_name} | Class: {class_name} | Boxes: {n_boxes}"
        cv2.putText(self.display_image, info, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _load_image(self, idx: int) -> None:
        """Load image at index."""
        img_path = self.image_files[idx]
        self.current_image = cv2.imread(str(img_path))
        
        if self.current_image is None:
            logger.error(f"Failed to load image: {img_path}")
            return
        
        self.img_height, self.img_width = self.current_image.shape[:2]
        
        # Calculate scale to fit in max window size
        max_w, max_h = self.max_window_size
        scale_w = max_w / self.img_width
        scale_h = max_h / self.img_height
        self.scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        # Load existing annotations if any
        label_path = self.labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            annotation = Annotation.load_label(str(img_path), str(label_path))
            self.state.boxes = annotation.boxes
        else:
            self.state.boxes = []
        
        self._draw()
    
    def _save_current(self) -> None:
        """Save annotations for current image."""
        if not self.image_files:
            return
        
        img_path = self.image_files[self.current_idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")
        
        annotation = Annotation(
            image_path=str(img_path),
            boxes=self.state.boxes,
            img_width=self.img_width,
            img_height=self.img_height
        )
        annotation.save_label(str(label_path))
        logger.info(f"Saved {len(self.state.boxes)} boxes to {label_path}")
    
    def _next_image(self) -> None:
        """Go to next image."""
        self._save_current()
        self.current_idx = (self.current_idx + 1) % len(self.image_files)
        self._load_image(self.current_idx)
    
    def _prev_image(self) -> None:
        """Go to previous image."""
        self._save_current()
        self.current_idx = (self.current_idx - 1) % len(self.image_files)
        self._load_image(self.current_idx)
    
    def _cycle_class(self) -> None:
        """Cycle to next class."""
        self.state.class_id = (self.state.class_id + 1) % len(self.class_names)
        self._draw()
    
    def run(self) -> None:
        """Run the annotation loop."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        self._load_image(self.current_idx)
        
        print("\n=== Image Annotator ===")
        print("Controls:")
        print("  Left click + drag: Draw box")
        print("  Right click: Delete last box")
        print("  n/Space: Next image")
        print("  p/Backspace: Previous image")
        print("  s: Save")
        print("  c: Change class")
        print("  u: Undo last box")
        print("  r: Reset all boxes")
        print("  q/Esc: Quit")
        print("=" * 25 + "\n")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # q or Esc
                self._save_current()
                break
            
            elif key == ord('n') or key == 32:  # n or Space
                self._next_image()
            
            elif key == ord('p') or key == 8:  # p or Backspace
                self._prev_image()
            
            elif key == ord('s'):
                self._save_current()
            
            elif key == ord('c'):
                self._cycle_class()
            
            elif key == ord('u'):
                if self.state.boxes:
                    self.state.boxes.pop()
                    self._draw()
            
            elif key == ord('r'):
                self.state.boxes = []
                self._draw()
        
        cv2.destroyAllWindows()
        logger.info("Annotation session ended")
    
    def get_progress(self) -> Dict:
        """Get annotation progress."""
        annotated = 0
        total_boxes = 0
        
        for img_path in self.image_files:
            label_path = self.labels_dir / (img_path.stem + ".txt")
            if label_path.exists() and label_path.stat().st_size > 0:
                annotated += 1
                with open(label_path) as f:
                    total_boxes += len([l for l in f if l.strip()])
        
        return {
            "total_images": len(self.image_files),
            "annotated": annotated,
            "remaining": len(self.image_files) - annotated,
            "total_boxes": total_boxes,
            "progress_percent": (annotated / len(self.image_files) * 100) if self.image_files else 0
        }


def annotate(
    images_dir: str,
    labels_dir: str,
    class_names: List[str] = None
) -> None:
    """
    Quick function to start annotating images.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory to save labels
        class_names: List of class names
    """
    annotator = ImageAnnotator(images_dir, labels_dir, class_names)
    annotator.run()
