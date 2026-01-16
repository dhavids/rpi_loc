#!/usr/bin/env python3
"""
YOLO Trainer for TurtleBot Detection

Handles training and fine-tuning of YOLO models for TurtleBot detection.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil
import yaml

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed. Install with: pip install ultralytics")


@dataclass
class TrainingConfig:
    """Configuration for YOLO training."""
    # Model
    base_model: str = "yolov8n.pt"  # Base model to fine-tune
    
    # Dataset
    data_yaml: Optional[str] = None  # Path to data.yaml
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    patience: int = 50  # Early stopping patience
    
    # Optimizer
    optimizer: str = "auto"  # SGD, Adam, AdamW, auto
    lr0: float = 0.01  # Initial learning rate
    lrf: float = 0.01  # Final learning rate factor
    momentum: float = 0.937
    weight_decay: float = 0.0005
    
    # Augmentation
    augment: bool = True
    hsv_h: float = 0.015  # HSV-Hue augmentation
    hsv_s: float = 0.7    # HSV-Saturation augmentation
    hsv_v: float = 0.4    # HSV-Value augmentation
    degrees: float = 0.0  # Rotation augmentation
    translate: float = 0.1
    scale: float = 0.5
    fliplr: float = 0.5   # Horizontal flip probability
    flipud: float = 0.0   # Vertical flip probability
    mosaic: float = 1.0   # Mosaic augmentation probability
    mixup: float = 0.0    # Mixup augmentation probability
    
    # Output
    project: str = "runs/train"
    name: str = "turtlebot"
    exist_ok: bool = True
    
    # Hardware
    device: str = "auto"  # cuda, cpu, mps, auto
    workers: int = 8
    
    # Validation
    val: bool = True
    save_period: int = -1  # Save checkpoint every N epochs (-1 = disabled)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for YOLO."""
        return {
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.image_size,
            "patience": self.patience,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "augment": self.augment,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "fliplr": self.fliplr,
            "flipud": self.flipud,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "project": self.project,
            "name": self.name,
            "exist_ok": self.exist_ok,
            "device": None if self.device == "auto" else self.device,
            "workers": self.workers,
            "val": self.val,
            "save_period": self.save_period,
        }


class YOLOTrainer:
    """
    Trainer for YOLO models on TurtleBot detection.
    
    Supports:
    - Training from scratch
    - Fine-tuning pretrained models
    - Transfer learning from COCO weights
    
    Example:
        >>> trainer = YOLOTrainer()
        >>> config = TrainingConfig(epochs=50, batch_size=8)
        >>> trainer.train("path/to/data.yaml", config)
        >>> trainer.export("best.pt", format="onnx")
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            model_path: Path to existing model to continue training.
                       If None, will use base_model from config.
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
        
        self.model: Optional[YOLO] = None
        self.model_path = model_path
        self.results = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load a YOLO model."""
        path = Path(model_path)
        if not path.exists() and not model_path.endswith(".pt"):
            # Try adding .pt extension
            path = Path(model_path + ".pt")
        
        logger.info(f"Loading model from {path}")
        self.model = YOLO(str(path))
        self.model_path = str(path)
    
    def train(
        self,
        data_yaml: str,
        config: Optional[TrainingConfig] = None,
        resume: bool = False
    ) -> Dict:
        """
        Train the YOLO model.
        
        Args:
            data_yaml: Path to dataset YAML file
            config: Training configuration
            resume: Whether to resume from last checkpoint
            
        Returns:
            Training results dictionary
        """
        if config is None:
            config = TrainingConfig()
        
        # Load base model if not already loaded
        if self.model is None:
            logger.info(f"Loading base model: {config.base_model}")
            self.model = YOLO(config.base_model)
        
        # Verify data.yaml exists
        data_path = Path(data_yaml)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {data_yaml}")
        
        logger.info(f"Starting training with config: epochs={config.epochs}, "
                   f"batch_size={config.batch_size}, image_size={config.image_size}")
        
        # Get training parameters
        train_params = config.to_dict()
        train_params["data"] = str(data_path)
        train_params["resume"] = resume
        
        # Train
        self.results = self.model.train(**train_params)
        
        # Get best model path from results
        if self.results and hasattr(self.results, 'save_dir'):
            best_path = Path(self.results.save_dir) / "weights" / "best.pt"
            if best_path.exists():
                self.model_path = str(best_path)
                logger.info(f"Best model saved to: {best_path}")
        
        return self._parse_results()
    
    def finetune(
        self,
        data_yaml: str,
        base_model: str = "yolov8n.pt",
        epochs: int = 50,
        freeze_layers: int = 0,
        **kwargs
    ) -> Dict:
        """
        Fine-tune a pretrained model on TurtleBot data.
        
        This is useful when you have limited training data.
        
        Args:
            data_yaml: Path to dataset YAML file
            base_model: Pretrained model to fine-tune
            epochs: Number of training epochs
            freeze_layers: Number of backbone layers to freeze (0 = none)
            **kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        config = TrainingConfig(
            base_model=base_model,
            epochs=epochs,
            **kwargs
        )
        
        # Load pretrained model
        logger.info(f"Fine-tuning from {base_model}")
        self.model = YOLO(base_model)
        
        # Freeze layers if specified
        if freeze_layers > 0:
            logger.info(f"Freezing first {freeze_layers} layers")
            self._freeze_layers(freeze_layers)
        
        return self.train(data_yaml, config)
    
    def _freeze_layers(self, num_layers: int) -> None:
        """Freeze the first N layers of the model."""
        if self.model is None:
            return
            
        for i, (name, param) in enumerate(self.model.model.named_parameters()):
            if i < num_layers:
                param.requires_grad = False
                logger.debug(f"Frozen: {name}")
    
    def validate(self, data_yaml: Optional[str] = None) -> Dict:
        """
        Validate the model on test/validation data.
        
        Args:
            data_yaml: Path to dataset YAML file. If None, uses training data.
            
        Returns:
            Validation metrics dictionary
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() or train() first.")
        
        kwargs = {}
        if data_yaml:
            kwargs["data"] = data_yaml
        
        results = self.model.val(**kwargs)
        
        return {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }
    
    def predict(
        self,
        source: str,
        confidence: float = 0.5,
        save: bool = True
    ) -> List[Dict]:
        """
        Run inference on images/video.
        
        Args:
            source: Path to image, video, or directory
            confidence: Minimum confidence threshold
            save: Whether to save annotated results
            
        Returns:
            List of detection results
        """
        if self.model is None:
            raise ValueError("No model loaded.")
        
        results = self.model.predict(
            source=source,
            conf=confidence,
            save=save
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                det = {
                    "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                    "confidence": float(boxes.conf[i].cpu().numpy()),
                    "class_id": int(boxes.cls[i].cpu().numpy()),
                    "class_name": result.names[int(boxes.cls[i])]
                }
                detections.append(det)
        
        return detections
    
    def export(
        self,
        format: str = "onnx",
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Export model to different formats.
        
        Args:
            format: Export format (onnx, torchscript, tflite, etc.)
            output_path: Output path (optional)
            **kwargs: Additional export parameters
            
        Returns:
            Path to exported model
        """
        if self.model is None:
            raise ValueError("No model loaded.")
        
        logger.info(f"Exporting model to {format} format")
        
        export_path = self.model.export(format=format, **kwargs)
        
        if output_path and export_path:
            shutil.copy(export_path, output_path)
            return output_path
        
        return str(export_path)
    
    def _parse_results(self) -> Dict:
        """Parse training results into a dictionary."""
        if self.results is None:
            return {}
        
        return {
            "best_epoch": getattr(self.results, "best_epoch", None),
            "best_fitness": getattr(self.results, "best_fitness", None),
            "model_path": self.model_path,
        }
    
    @staticmethod
    def create_data_yaml(
        train_path: str,
        val_path: str,
        class_names: List[str],
        output_path: str,
        test_path: Optional[str] = None
    ) -> str:
        """
        Create a YOLO data.yaml file.
        
        Args:
            train_path: Path to training images
            val_path: Path to validation images
            class_names: List of class names
            output_path: Where to save data.yaml
            test_path: Optional path to test images
            
        Returns:
            Path to created data.yaml
        """
        data = {
            "path": str(Path(output_path).parent.absolute()),
            "train": train_path,
            "val": val_path,
            "names": {i: name for i, name in enumerate(class_names)}
        }
        
        if test_path:
            data["test"] = test_path
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        
        logger.info(f"Created data.yaml at {output}")
        return str(output)


def quick_train(
    data_dir: str,
    epochs: int = 100,
    model_size: str = "n"
) -> str:
    """
    Quick training function for TurtleBot detection.
    
    Args:
        data_dir: Directory containing train/val folders with images and labels
        epochs: Number of training epochs
        model_size: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
        
    Returns:
        Path to best trained model
    """
    # Validate model size
    valid_sizes = ["n", "s", "m", "l", "x"]
    if model_size not in valid_sizes:
        raise ValueError(f"model_size must be one of {valid_sizes}")
    
    base_model = f"yolov8{model_size}.pt"
    
    # Check for data.yaml
    data_path = Path(data_dir)
    yaml_path = data_path / "data.yaml"
    
    if not yaml_path.exists():
        # Try to create data.yaml automatically
        train_path = data_path / "train" / "images"
        val_path = data_path / "val" / "images"
        
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"Expected train/images and val/images folders in {data_dir}, "
                "or a data.yaml file"
            )
        
        YOLOTrainer.create_data_yaml(
            train_path=str(train_path.relative_to(data_path)),
            val_path=str(val_path.relative_to(data_path)),
            class_names=["turtlebot"],
            output_path=str(yaml_path)
        )
    
    # Train
    config = TrainingConfig(
        base_model=base_model,
        epochs=epochs,
        name=f"turtlebot_{model_size}"
    )
    
    trainer = YOLOTrainer()
    trainer.train(str(yaml_path), config)
    
    return trainer.model_path
