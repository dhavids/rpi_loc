"""
YOLO Training and Fine-tuning Utilities for TurtleBot Detection

This module provides tools for:
- Collecting and annotating training images
- Creating YOLO-format datasets
- Training and fine-tuning YOLO models
- Evaluating model performance
"""

from .trainer import YOLOTrainer, TrainingConfig
from .dataset import DatasetManager, Annotation, BoundingBox
from .annotator import ImageAnnotator
from .collector import ImageCollector, StreamCollector, CollectorConfig

__all__ = [
    "YOLOTrainer",
    "TrainingConfig", 
    "DatasetManager",
    "Annotation",
    "BoundingBox",
    "ImageAnnotator",
    "ImageCollector",
    "StreamCollector",
    "CollectorConfig",
]
