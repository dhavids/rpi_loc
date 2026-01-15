# YOLO TurtleBot Detection Training

Tools for training custom YOLO models to detect TurtleBots.

## Installation

```bash
pip install ultralytics opencv-python pyyaml
```

## Quick Start

### 1. Collect Training Images

**From RPi camera stream (recommended):**
```bash
# Interactive mode - press Space to capture
python -m rpi_loc.src.models.yolo.cli collect --stream --host 100.99.98.1 --output data/images

# Auto-capture for 60 seconds at 2s intervals
python -m rpi_loc.src.models.yolo.cli collect --stream --duration 60 --interval 2 --output data/images
```

**From local camera:**
```bash
python -m rpi_loc.src.models.yolo.cli collect --camera 0 --output data/images
```

**From video:**
```bash
python -m rpi_loc.src.models.yolo.cli collect --video recording.mp4 --output data/images --every-n 30
```

### 2. Annotate Images

```bash
python -m rpi_loc.src.models.yolo.cli annotate --images data/images --labels data/labels
```

**Controls:**
- Left click + drag: Draw bounding box
- Right click: Delete last box
- Space/n: Next image
- Backspace/p: Previous image
- c: Change class
- s: Save
- q: Quit

### 3. Create Dataset

```bash
python -m rpi_loc.src.models.yolo.cli create-dataset \
    --images data/images \
    --labels data/labels \
    --output data/dataset \
    --train-ratio 0.8 \
    --val-ratio 0.15 \
    --test-ratio 0.05
```

### 4. Train Model

```bash
python -m rpi_loc.src.models.yolo.cli train \
    --data data/dataset \
    --epochs 100 \
    --batch-size 16 \
    --base-model yolov8n.pt
```

Or fine-tune:
```bash
python -m rpi_loc.src.models.yolo.cli finetune \
    --data data/dataset \
    --epochs 50 \
    --base-model yolov8n.pt \
    --freeze-layers 10
```

### 5. Evaluate Model

```bash
python -m rpi_loc.src.models.yolo.cli evaluate \
    --model runs/train/turtlebot/weights/best.pt \
    --data data/dataset
```

### 6. Use Trained Model

```python
from rpi_loc.utils.img_proc.image_parser import create_parser

parser = create_parser(
    model_path="runs/train/turtlebot/weights/best.pt",
    confidence_threshold=0.5
)

result = parser.process_frame(frame)
for tb in result.turtlebots:
    print(f"{tb.name}: ({tb.x:.2f}, {tb.y:.2f})m")
```

## Python API

### Training

```python
from rpi_loc.src.models.yolo import YOLOTrainer, TrainingConfig

config = TrainingConfig(
    epochs=100,
    batch_size=16,
    image_size=640,
    base_model="yolov8n.pt"
)

trainer = YOLOTrainer()
trainer.train("data/dataset/data.yaml", config)

# Export to ONNX
trainer.export(format="onnx", output_path="turtlebot.onnx")
```

### Dataset Management

```python
from rpi_loc.src.models.yolo import DatasetManager, BoundingBox

# Create dataset
manager = DatasetManager("data/dataset", class_names=["turtlebot"])
manager.create_structure()

# Add annotated images
box = BoundingBox.from_xyxy(0, 100, 100, 200, 200, 640, 480)
manager.add_image("image.jpg", [box], split="train")

# Split existing data
manager.split_dataset(
    images_folder="raw/images",
    labels_folder="raw/labels",
    train_ratio=0.8
)
```

### Image Collection

**From RPi camera stream:**
```python
from rpi_loc.src.models.yolo import StreamCollector, CollectorConfig

config = CollectorConfig(
    output_dir="collected_images",
    auto_capture_interval=2.0
)

# Connect to RPi camera stream
collector = StreamCollector(
    host="100.99.98.1",  # RPi Tailscale IP
    port=5000,
    config=config
)

# Interactive mode (press Space to capture)
collector.collect_interactive()

# Or auto-collect for 60 seconds
collector.collect_timed(duration=60, interval=2.0)
```

**From local camera:**
```python
from rpi_loc.src.models.yolo import ImageCollector, CollectorConfig

config = CollectorConfig(
    output_dir="collected_images",
    auto_capture_interval=2.0  # Every 2 seconds
)

collector = ImageCollector(source=0, config=config)
collector.collect_interactive()
```

## Model Sizes

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolov8n.pt | Nano | Fastest | Lower |
| yolov8s.pt | Small | Fast | Good |
| yolov8m.pt | Medium | Medium | Better |
| yolov8l.pt | Large | Slow | High |
| yolov8x.pt | XLarge | Slowest | Highest |

For Raspberry Pi, use `yolov8n.pt` or `yolov8s.pt`.

## Tips for Good Training Data

1. **Variety**: Capture TurtleBots from different angles, distances, and lighting
2. **Background**: Include varied backgrounds similar to deployment environment
3. **Quantity**: Aim for 100+ images minimum, 500+ for better results
4. **Balance**: Ensure similar number of images per TurtleBot (if using multiple classes)
5. **Occlusion**: Include some partially occluded TurtleBots
6. **Motion blur**: Include some images with motion blur if robots move fast

## Export Formats

```bash
# ONNX (recommended for deployment)
python -m rpi_loc.src.models.yolo.cli export --model best.pt --format onnx

# TensorRT (for NVIDIA GPUs)
python -m rpi_loc.src.models.yolo.cli export --model best.pt --format engine

# TFLite (for edge devices)
python -m rpi_loc.src.models.yolo.cli export --model best.pt --format tflite

# CoreML (for Apple devices)
python -m rpi_loc.src.models.yolo.cli export --model best.pt --format coreml
```
