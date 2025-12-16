# Traffic Heatmap - Multi-Camera Vision System

Real-time customer traffic analysis system that processes multiple RTSP camera streams to generate unified heatmaps on a 2D store floor plan using computer vision and homography transformation.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [Calibration System](#calibration-system)
9. [Development Roadmap](#development-roadmap)
10. [License](#license)

---

## Overview

This system transforms multiple camera feeds into actionable retail analytics by:

- Detecting people in multiple simultaneous RTSP camera streams using YOLOv8
- Transforming camera coordinates to a unified 2D store map using homography matrices
- Generating real-time heatmaps showing customer traffic patterns
- Exporting JSON data for visualization in web/mobile interfaces

## Features

### Current Implementation (Task 1 - Foundations)

- [x] RTSP camera stream connection and testing
- [x] YOLOv8 person detection with configurable models
- [x] Multi-camera calibration system with Qt GUI
- [x] Homography matrix calculation and storage
- [x] Point-to-point mapping between camera view and floor plan

### Planned Features

- [ ] Multi-camera simultaneous processing
- [ ] Real-time heatmap generation
- [ ] Web dashboard for visualization
- [ ] Historical data analysis
- [ ] API endpoints for integration

## Use Cases

- Retail store traffic analysis
- Queue management optimization
- Store layout effectiveness
- Peak hours identification
- Customer behavior insights

---

## Project Structure

```
Traffic-Heatmap/
|-- calibration_data/           # Stored calibration files and snapshots
|   |-- snapshots/              # Camera frame snapshots
|   |-- *.json                  # Calibration and homography matrix files
|
|-- calibration_system/         # Qt-based calibration application
|   |-- __init__.py
|   |-- calibration_app.py      # Main calibration GUI application
|   |-- camera_manager.py       # RTSP camera connection management
|   |-- config.py               # Configuration data classes
|   |-- homography.py           # Homography matrix calculations
|   |-- storage.py              # JSON storage for calibration data
|   |-- widgets.py              # Custom Qt widgets for point selection
|
|-- models/                     # YOLO model management
|   |-- __init__.py
|   |-- scripts/
|   |   |-- __init__.py
|   |   |-- config.py           # Model configuration
|   |   |-- yolo_loader.py      # YOLO model loading utilities
|   |-- weights/                # Model weight files (yolov8s.pt, etc.)
|
|-- src/                        # Source scripts
|   |-- test_rtsp_connection.py # RTSP connection testing utility
|   |-- yolo_person_detection.py # Real-time person detection script
|   |-- output/                 # Detection output files
|
|-- .env                        # Environment configuration (not in repo)
|-- .env.example                # Environment template
|-- requirements.txt            # Python dependencies
|-- README.md                   # This file
|-- LICENSE                     # Project license
```

---

## Requirements

### System Requirements

- Python 3.8 or higher
- Linux/Windows/macOS
- Network access to RTSP cameras
- GPU recommended for real-time processing (optional)

### Python Dependencies

```
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
python-dotenv>=1.0.0
PyQt6>=6.5.0
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ElSangour/Traffic-Heatmap.git
cd Traffic-Heatmap
```

### 2. Create Virtual Environment

```bash
python -m venv .heatmap_env
source .heatmap_env/bin/activate  # Linux/macOS
# or
.heatmap_env\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLO Model

The YOLOv8 model will be downloaded automatically on first run, or you can manually download it:

```bash
# Using ultralytics CLI
yolo export model=yolov8s.pt format=pytorch

# Or download directly and place in models/weights/
```

---

## Configuration

### Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your camera settings:

```dotenv
# RTSP Camera Configuration
RTSP_URL=rtsp://username:password@camera_ip:554/stream
RTSP_USERNAME=admin
RTSP_PASSWORD=your_password
CAMERA_IP=192.168.1.100

# YOLOv8 Model Configuration
YOLO_MODEL=yolov8s.pt
CONFIDENCE_THRESHOLD=0.5

# Frame Processing
FRAME_WIDTH=640
FRAME_HEIGHT=480

# Output Settings
OUTPUT_DIR=./output
DISPLAY_WINDOW=true
```

### Available YOLO Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolov8n.pt | Nano | Fastest | Good |
| yolov8s.pt | Small | Fast | Better |
| yolov8m.pt | Medium | Medium | High |
| yolov8l.pt | Large | Slow | Higher |
| yolov8x.pt | XLarge | Slowest | Highest |

---

## Usage

### Test RTSP Connection

Verify camera connectivity before running detection:

```bash
python src/test_rtsp_connection.py
```

### Run Person Detection

Start real-time person detection on RTSP stream:

```bash
python src/yolo_person_detection.py
```

Controls:
- Press `q` to quit
- Press `s` to save current frame

---

## Calibration System

The calibration system provides a graphical interface for mapping camera views to floor plans using homography transformation.

### Launch Calibration App

```bash
python calibration_system/calibration_app.py
```

### Calibration Workflow

1. **Setup Configuration**
   - Enter store name
   - Provide RTSP URL template with `{camera_id}` placeholder
   - Enter camera IDs to calibrate (comma-separated)
   - Load your store floor plan image

2. **Point Selection**
   - Select corresponding points on camera view and floor plan
   - Minimum 4 points required (more points = better accuracy)
   - Points should cover the visible floor area

3. **Calculate Homography**
   - Click "Calculate Homography" to compute transformation matrix
   - Check reprojection error (lower is better, aim for < 5px)

4. **Save Calibration**
   - Calibration data saved to `calibration_data/`
   - Homography matrices exported for each camera

### Calibration Tips

- Use clearly identifiable floor markers or corners
- Distribute points across the entire visible floor area
- Avoid selecting points on vertical surfaces
- Use at least 6-8 points for better accuracy
- Verify calibration by checking mapped coordinates

### Output Files

Calibration files are saved with the store name:

```
calibration_data/
|-- calibration_{store_name}_{timestamp}.json    # Full calibration data
|-- homography_matrices_{store_name}.json        # Matrix-only export
|-- snapshots/                                   # Camera frame captures
```

---

## Development Roadmap

### Task 1: Foundations and Single Camera Prototype [CURRENT]
- [x] Environment setup and dependencies
- [x] RTSP stream connection
- [x] YOLOv8 person detection
- [x] Multi-camera calibration system
- [x] Homography matrix generation

### Task 2: Multi-Camera System
- [ ] Simultaneous multi-camera processing
- [ ] Coordinate transformation pipeline
- [ ] Unified detection aggregation

### Task 3: Heatmap Generation
- [ ] Real-time heatmap rendering
- [ ] Configurable time windows
- [ ] Heat decay algorithms

### Task 4: Data Export and Visualization
- [ ] JSON/CSV data export
- [ ] Web dashboard
- [ ] Historical analysis tools

### Task 5: Optimization and Deployment
- [ ] Performance optimization
- [ ] Docker containerization
- [ ] Production deployment

---

## API Reference

### YOLOPersonDetector

```python
from models import YOLOPersonDetector

# Initialize detector
detector = YOLOPersonDetector(model_name="yolov8s.pt", confidence=0.5)

# Detect persons in frame
detections = detector.detect_persons(frame)

# Each detection contains:
# - bbox: (x1, y1, x2, y2)
# - confidence: float
# - class_id: int (0 for person)
```

### HomographyCalculator

```python
from calibration_system.homography import HomographyCalculator

calc = HomographyCalculator()

# Calculate homography from point pairs
result = calc.calculate_homography(camera_points, plan_points)

if result.is_valid:
    # Transform point from camera to plan
    plan_point = calc.transform_point(camera_point, result.matrix)
```

---

## Troubleshooting

### RTSP Connection Issues

```
[ERROR] Cannot connect to RTSP stream
```

- Verify camera IP and credentials
- Check network connectivity
- Ensure RTSP port is accessible
- Try the URL in VLC player first

### YOLO Model Not Found

```
[ERROR] Model not found
```

- Run detection script once to auto-download
- Or manually download to `models/weights/`

### Qt Display Issues

```
qt.qpa.xcb: could not connect to display
```

- Ensure X11 forwarding if using SSH
- Set `DISPLAY` environment variable
- Use `export QT_QPA_PLATFORM=offscreen` for headless

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

---

## License

This project is licensed under the terms specified in the LICENSE file.

---

## Contact

For questions or support, please open an issue on the GitHub repository.