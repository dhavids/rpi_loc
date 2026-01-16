#!/usr/bin/env python3
"""
Dataset Management for YOLO Training

Handles dataset creation, organization, and manipulation for
training YOLO models on TurtleBot detection.
"""

import logging
import shutil
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parents[3]))
from utils.core.setup_logging import get_named_logger

logger = get_named_logger("dataset", __name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class BoundingBox:
    """Bounding box in YOLO format (normalized)."""
    class_id: int
    x_center: float  # Normalized (0-1)
    y_center: float  # Normalized (0-1)
    width: float     # Normalized (0-1)
    height: float    # Normalized (0-1)
    
    def to_yolo_line(self) -> str:
        """Convert to YOLO label format line."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"
    
    @classmethod
    def from_yolo_line(cls, line: str) -> "BoundingBox":
        """Parse from YOLO label format line."""
        parts = line.strip().split()
        return cls(
            class_id=int(parts[0]),
            x_center=float(parts[1]),
            y_center=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4])
        )
    
    @classmethod
    def from_xyxy(
        cls, 
        class_id: int,
        x1: int, y1: int, x2: int, y2: int,
        img_width: int, img_height: int
    ) -> "BoundingBox":
        """Create from pixel coordinates (x1, y1, x2, y2)."""
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        return cls(
            class_id=class_id,
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height
        )
    
    def to_xyxy(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert to pixel coordinates (x1, y1, x2, y2)."""
        w = self.width * img_width
        h = self.height * img_height
        cx = self.x_center * img_width
        cy = self.y_center * img_height
        
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        
        return (x1, y1, x2, y2)


@dataclass
class Annotation:
    """Annotation for a single image."""
    image_path: str
    boxes: List[BoundingBox] = field(default_factory=list)
    img_width: Optional[int] = None
    img_height: Optional[int] = None
    
    def save_label(self, label_path: str) -> None:
        """Save annotations to YOLO label file."""
        path = Path(label_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            for box in self.boxes:
                f.write(box.to_yolo_line() + "\n")
    
    @classmethod
    def load_label(cls, image_path: str, label_path: str) -> "Annotation":
        """Load annotation from YOLO label file."""
        boxes = []
        
        label_file = Path(label_path)
        if label_file.exists():
            with open(label_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        boxes.append(BoundingBox.from_yolo_line(line))
        
        return cls(image_path=image_path, boxes=boxes)


class DatasetManager:
    """
    Manages YOLO-format datasets for TurtleBot detection.
    
    Dataset structure:
        dataset/
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   │   ├── img001.jpg
        │   │   └── ...
        │   └── labels/
        │       ├── img001.txt
        │       └── ...
        ├── val/
        │   ├── images/
        │   └── labels/
        └── test/  (optional)
            ├── images/
            └── labels/
    """
    
    def __init__(
        self,
        root_dir: str,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize dataset manager.
        
        Args:
            root_dir: Root directory for the dataset
            class_names: List of class names (default: ["turtlebot"])
        """
        self.root_dir = Path(root_dir)
        self.class_names = class_names or ["turtlebot"]
        
        # Create directory structure
        self.train_images = self.root_dir / "train" / "images"
        self.train_labels = self.root_dir / "train" / "labels"
        self.val_images = self.root_dir / "val" / "images"
        self.val_labels = self.root_dir / "val" / "labels"
        self.test_images = self.root_dir / "test" / "images"
        self.test_labels = self.root_dir / "test" / "labels"
        
        self.data_yaml = self.root_dir / "data.yaml"
    
    def create_structure(self) -> None:
        """Create the dataset directory structure."""
        for path in [self.train_images, self.train_labels,
                     self.val_images, self.val_labels]:
            path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created dataset structure at {self.root_dir}")
    
    def create_data_yaml(self) -> str:
        """Create the data.yaml file."""
        data = {
            "path": str(self.root_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "names": {i: name for i, name in enumerate(self.class_names)}
        }
        
        # Add test if it exists
        if self.test_images.exists():
            data["test"] = "test/images"
        
        with open(self.data_yaml, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        
        logger.info(f"Created data.yaml at {self.data_yaml}")
        return str(self.data_yaml)
    
    def add_image(
        self,
        image_path: str,
        boxes: List[BoundingBox],
        split: str = "train",
        copy: bool = True
    ) -> str:
        """
        Add an image with annotations to the dataset.
        
        Args:
            image_path: Path to the image file
            boxes: List of bounding boxes for the image
            split: Dataset split (train, val, test)
            copy: Whether to copy the image (if False, moves it)
            
        Returns:
            Path to the image in the dataset
        """
        src_path = Path(image_path)
        
        if split == "train":
            img_dir = self.train_images
            label_dir = self.train_labels
        elif split == "val":
            img_dir = self.val_images
            label_dir = self.val_labels
        elif split == "test":
            img_dir = self.test_images
            label_dir = self.test_labels
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Ensure directories exist
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy/move image
        dst_img = img_dir / src_path.name
        if copy:
            shutil.copy2(src_path, dst_img)
        else:
            shutil.move(src_path, dst_img)
        
        # Save label
        label_path = label_dir / (src_path.stem + ".txt")
        annotation = Annotation(image_path=str(dst_img), boxes=boxes)
        annotation.save_label(str(label_path))
        
        return str(dst_img)
    
    def add_images_from_folder(
        self,
        image_folder: str,
        labels_folder: str,
        split: str = "train"
    ) -> int:
        """
        Add images from a folder with existing YOLO labels.
        
        Args:
            image_folder: Folder containing images
            labels_folder: Folder containing YOLO label files
            split: Dataset split (train, val, test)
            
        Returns:
            Number of images added
        """
        img_folder = Path(image_folder)
        lbl_folder = Path(labels_folder)
        
        count = 0
        for img_path in img_folder.glob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            
            # Find corresponding label
            label_path = lbl_folder / (img_path.stem + ".txt")
            
            if label_path.exists():
                annotation = Annotation.load_label(str(img_path), str(label_path))
                self.add_image(str(img_path), annotation.boxes, split=split)
                count += 1
        
        logger.info(f"Added {count} images to {split} split")
        return count
    
    def split_dataset(
        self,
        images_folder: str,
        labels_folder: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        shuffle: bool = True,
        seed: int = 42
    ) -> Dict[str, int]:
        """
        Split a folder of images into train/val/test sets.
        
        Args:
            images_folder: Folder containing images
            labels_folder: Folder containing YOLO labels
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            shuffle: Whether to shuffle before splitting
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with counts per split
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Ratios must sum to 1.0")
        
        img_folder = Path(images_folder)
        lbl_folder = Path(labels_folder)
        
        # Get all images with labels
        valid_images = []
        for img_path in img_folder.glob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            label_path = lbl_folder / (img_path.stem + ".txt")
            if label_path.exists():
                valid_images.append((img_path, label_path))
        
        if not valid_images:
            raise ValueError("No valid image-label pairs found")
        
        # Shuffle
        if shuffle:
            random.seed(seed)
            random.shuffle(valid_images)
        
        # Split
        n = len(valid_images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_set = valid_images[:n_train]
        val_set = valid_images[n_train:n_train + n_val]
        test_set = valid_images[n_train + n_val:]
        
        # Create dataset structure
        self.create_structure()
        
        # Add images to each split
        for img_path, label_path in train_set:
            annotation = Annotation.load_label(str(img_path), str(label_path))
            self.add_image(str(img_path), annotation.boxes, split="train")
        
        for img_path, label_path in val_set:
            annotation = Annotation.load_label(str(img_path), str(label_path))
            self.add_image(str(img_path), annotation.boxes, split="val")
        
        if test_set:
            self.test_images.mkdir(parents=True, exist_ok=True)
            self.test_labels.mkdir(parents=True, exist_ok=True)
            for img_path, label_path in test_set:
                annotation = Annotation.load_label(str(img_path), str(label_path))
                self.add_image(str(img_path), annotation.boxes, split="test")
        
        # Create data.yaml
        self.create_data_yaml()
        
        counts = {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set)
        }
        
        logger.info(f"Dataset split complete: {counts}")
        return counts
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            "train_images": len(list(self.train_images.glob("*"))) if self.train_images.exists() else 0,
            "val_images": len(list(self.val_images.glob("*"))) if self.val_images.exists() else 0,
            "test_images": len(list(self.test_images.glob("*"))) if self.test_images.exists() else 0,
            "classes": self.class_names,
            "num_classes": len(self.class_names)
        }
        
        # Count annotations
        train_boxes = 0
        if self.train_labels.exists():
            for label_file in self.train_labels.glob("*.txt"):
                with open(label_file) as f:
                    train_boxes += len([l for l in f if l.strip()])
        stats["train_annotations"] = train_boxes
        
        return stats
    
    def verify(self) -> Dict[str, List[str]]:
        """
        Verify dataset integrity.
        
        Returns:
            Dictionary with lists of issues found
        """
        issues = {
            "missing_labels": [],
            "missing_images": [],
            "empty_labels": [],
            "invalid_format": []
        }
        
        # Check train
        for img_path in self.train_images.glob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            label_path = self.train_labels / (img_path.stem + ".txt")
            if not label_path.exists():
                issues["missing_labels"].append(str(img_path))
            elif label_path.stat().st_size == 0:
                issues["empty_labels"].append(str(label_path))
        
        # Check val
        for img_path in self.val_images.glob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            label_path = self.val_labels / (img_path.stem + ".txt")
            if not label_path.exists():
                issues["missing_labels"].append(str(img_path))
        
        # Check data.yaml
        if not self.data_yaml.exists():
            issues["invalid_format"].append("data.yaml missing")
        
        return issues


def create_dataset(
    output_dir: str,
    class_names: Optional[List[str]] = None
) -> DatasetManager:
    """
    Create a new empty dataset.
    
    Args:
        output_dir: Directory to create dataset in
        class_names: List of class names
        
    Returns:
        DatasetManager instance
    """
    manager = DatasetManager(output_dir, class_names)
    manager.create_structure()
    manager.create_data_yaml()
    return manager
