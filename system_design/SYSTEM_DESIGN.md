# System Design and Architecture

Traffic Heatmap is a modular, multi-threaded computer vision system designed for real-time customer traffic analysis. This document outlines the system architecture, design decisions, and data flow.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Design Patterns](#design-patterns)
6. [Module Dependencies](#module-dependencies)
7. [Threading Model](#threading-model)
8. [Data Structures](#data-structures)
9. [Configuration Management](#configuration-management)
10. [Error Handling](#error-handling)

---

## System Overview

The Traffic Heatmap system is built on a modular architecture with clear separation of concerns:

```
RTSP Cameras
    |
    v
Camera Manager (RTSP Connection)
    |
    v
YOLO Detector (Person Detection)
    |
    v
Homography Transformer (Coordinate Transformation)
    |
    v
Detection Storage (JSON/Database)
    |
    v
Heatmap Generator (Visualization)
    |
    v
Qt Application (User Interface)
```

### Key Design Principles

- **Modularity**: Each component has a single responsibility
- **Concurrency**: Multi-threaded camera processing for real-time performance
- **Configurability**: Environment-based configuration for easy deployment
- **Extensibility**: Plugin-like architecture for adding new features
- **Robustness**: Comprehensive error handling and logging

---

## Architecture Diagram

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Qt Desktop Application                        │
│  ┌────────────────┬─────────────────┬───────────────────────┐   │
│  │ Calibration    │ Live Detection  │ Heatmap Visualization │   │
│  │ System         │ System          │ System                │   │
│  └────────────────┴─────────────────┴───────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         |                    |                        |
         v                    v                        v
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Calibration      │  │ Detection        │  │ Heatmap          │
│ Engine           │  │ Engine           │  │ Engine           │
│                  │  │                  │  │                  │
│- Point Select    │  │- Multi-Camera    │  │- Density Map     │
│- Homography      │  │- YOLO Detection  │  │- Gaussian Blur   │
│- Matrix Storage  │  │- Transformation  │  │- Colormap Apply  │
└──────────────────┘  │- JSON Export     │  │- PNG Save        │
                      └──────────────────┘  └──────────────────┘
         |                    |                        |
         v                    v                        v
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Calibration      │  │ Detection        │  │ Heatmap          │
│ Storage          │  │ Storage          │  │ Storage          │
│                  │  │                  │  │                  │
│calibration_*.json│  │detections_*.json │  │heatmap_*.png     │
│homography_*.json │  │                  │  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

### Data Flow Architecture

```
RTSP Streams → Camera Manager → Frame Queue → YOLO Detector
                                                    |
                                                    v
                                            Detection Results
                                                    |
                ┌───────────────────────────────────┴─────────────────────┐
                |                                                         |
                v                                                         v
        Foot Point Extraction                               Coordinate Transform
        (Bottom-center of bbox)                            (H @ camera_point)
                |                                                         |
                └───────────────────────────────────────────────────────┘
                                    |
                                    v
                            Detection Records
                    (timestamp, camera_id, coords, conf)
                                    |
                                    v
                            JSON Storage
                                    |
                                    v
                            Heatmap Generator
                                    |
                    ┌───────────────┴───────────────┐
                    |                               |
                    v                               v
            Density Histogram                 Floor Plan Image
                    |                               |
                    └───────────────┬───────────────┘
                                    v
                            Gaussian Smoothing
                                    |
                                    v
                            Colormap Application
                                    |
                                    v
                            Heatmap PNG Output
```

---

## Core Components

### 1. Camera Manager (`calibration_system/camera_manager.py`)

**Purpose**: Manage RTSP camera connections and frame capture

**Responsibilities**:
- Establish connections to RTSP streams
- Handle frame capture and buffering
- Manage camera lifecycle (connect, disconnect, reconnect)
- Provide configurable resolution and FPS

**Interface**:
```python
class CameraManager:
    def connect(camera_id: int, rtsp_url: str) -> bool
    def get_frame(camera_id: int) -> Frame
    def disconnect(camera_id: int) -> None
    def is_connected(camera_id: int) -> bool
    def get_properties(camera_id: int) -> CameraProperties
```

**Key Features**:
- Frame queuing for smooth processing
- Automatic reconnection on failure
- Resolution and FPS configuration
- Frame timestamp tracking

---

### 2. YOLO Detector (`models/scripts/yolo_loader.py`)

**Purpose**: Person detection using YOLOv8 neural network

**Responsibilities**:
- Load and initialize YOLO models
- Run inference on video frames
- Filter detections by class (person only)
- Track confidence scores

**Interface**:
```python
class YOLOPersonDetector:
    def __init__(model_name: str, confidence: float)
    def detect_persons(frame: ndarray) -> List[Detection]
    def detect_persons_batch(frames: List[ndarray]) -> List[List[Detection]]

class Detection:
    x1, y1, x2, y2: int          # Bounding box
    confidence: float             # Detection confidence
    class_id: int                 # Always 0 for person
```

**Key Features**:
- Model auto-download on first run
- Configurable confidence threshold
- Batch processing support
- Bounding box format: (x1, y1, x2, y2)

---

### 3. Homography Transformer (`calibration_system/homography.py`)

**Purpose**: Transform coordinates from camera space to floor plan space

**Responsibilities**:
- Calculate homography matrices from point pairs
- Transform individual points
- Validate transformation accuracy
- Handle batch transformations

**Interface**:
```python
class HomographyCalculator:
    def calculate_homography(
        camera_points: List[Point],
        plan_points: List[Point]
    ) -> HomographyResult
    
    def transform_point(
        point: Point,
        matrix: ndarray
    ) -> Point
    
    def calculate_reprojection_error(
        camera_points: List[Point],
        plan_points: List[Point],
        matrix: ndarray
    ) -> float
```

**Key Features**:
- OpenCV-based homography calculation
- RANSAC outlier rejection (optional)
- Error metrics for validation
- Batch point transformation

---

### 4. Detection Storage (`src/multi_camera_detector.py`)

**Purpose**: Store and manage detection data

**Responsibilities**:
- Collect detections from all cameras
- Calculate foot positions
- Transform to floor plan coordinates
- Export to JSON format

**Data Structure**:
```json
{
  "metadata": {
    "store_name": "mg_cite_olympique",
    "start_time": "2025-12-17T14:04:06",
    "duration_seconds": 60,
    "total_detections": 7972,
    "cameras": ["camera_1", "camera_2", ...]
  },
  "detections": [
    {
      "timestamp": "2025-12-17T14:04:06.123456",
      "camera_id": 1,
      "camera_name": "Camera 1",
      "camera_point": {"x": 320, "y": 480},
      "plan_point": {"x": 150, "y": 280},
      "confidence": 0.87
    }
  ]
}
```

---

### 5. Heatmap Generator (`src/heatmap_generator.py`)

**Purpose**: Generate visual heatmap from detection data

**Responsibilities**:
- Load detection JSON data
- Create 2D density histogram
- Apply Gaussian smoothing
- Apply colormap and overlay
- Save visualization

**Process**:
```
Detection Points
    |
    v
2D Histogram (binning by coordinate)
    |
    v
Gaussian Smoothing (cv2.GaussianBlur)
    |
    v
Colormap Application (cv2.applyColorMap)
    |
    v
Overlay on Floor Plan (weighted blend)
    |
    v
PNG Output
```

---

### 6. Qt Application (`main_app.py`)

**Purpose**: Unified graphical interface for all system components

**Responsibilities**:
- Provide user-friendly calibration interface
- Display live detection results
- Generate and visualize heatmaps
- Manage system configuration

**Tabs**:
1. **Calibration Tab** - Set up camera-to-plan mappings
2. **Live Detection Tab** - Run detection on multiple cameras
3. **Heatmap Tab** - Visualize detection data

---

## Data Flow

### Calibration Flow

```
User Starts App
    |
    v
Select "New Calibration"
    |
    v
Input Store Name & Floor Plan
    |
    v
Add Cameras (RTSP URL + IDs)
    |
    v
For Each Camera:
    ├─ Connect to RTSP Stream
    ├─ Display Live Feed
    ├─ User Selects Point Pairs (4+ minimum)
    ├─ Calculate Homography Matrix
    ├─ Validate (Reprojection Error < 5px)
    └─ Save Calibration Data
    |
    v
Export Homography Matrices (JSON)
    |
    v
Calibration Complete
```

### Detection Flow

```
Load Calibration Data
    |
    v
Connect to All Cameras
    |
    v
For Each Frame (Concurrent):
    ├─ Capture Frame from Camera
    ├─ Run YOLO Detection
    ├─ For Each Person Detected:
    │   ├─ Extract Foot Point (x_center, y_bottom)
    │   ├─ Transform Point (H @ foot_point)
    │   └─ Store Detection Record
    └─ Repeat Until Duration/Frame Limit
    |
    v
Export Detection JSON
    |
    v
Detection Complete
```

### Heatmap Generation Flow

```
Load Detection Data (JSON)
    |
    v
Load Floor Plan Image
    |
    v
Extract Plan Coordinates
    |
    v
Create Empty Histogram
    |
    v
Bin Detections into Histogram
    |
    v
Gaussian Smoothing
    |
    v
Normalize to 8-bit (0-255)
    |
    v
Apply Colormap
    |
    v
Overlay on Floor Plan (Alpha Blend)
    |
    v
Save as PNG
    |
    v
Heatmap Complete
```

---

## Design Patterns

### 1. Factory Pattern

**Location**: `models/scripts/yolo_loader.py`

```python
def load_yolo_model(model_name: str = None) -> YOLOPersonDetector:
    # Factory function - creates appropriate detector
    # Handles model download, caching
    return YOLOPersonDetector(model_name, confidence)
```

**Benefits**:
- Centralized model management
- Automatic download and caching
- Configuration-driven model selection

---

### 2. Strategy Pattern

**Location**: `src/heatmap_generator.py`

Different heatmap generation strategies:
- **Strategy 1**: Simple histogram + Gaussian blur
- **Strategy 2**: KDE (kernel density estimation)
- **Strategy 3**: Point density with decay

Current implementation uses Strategy 1 (simple and efficient).

---

### 3. Observer Pattern

**Location**: `main_app.py`

Qt signals/slots for event handling:
```python
# Calibration complete signal
calibration_complete = pyqtSignal(str)

# Detection update signal
detection_update = pyqtSignal(list)

# Heatmap ready signal
heatmap_ready = pyqtSignal(str)
```

---

### 4. Singleton Pattern

**Location**: `calibration_system/config.py`

Configuration loaded once, shared across application:
```python
config = CalibrationConfig.load("calibration_file.json")
# Reused throughout application
```

---

## Module Dependencies

```
main_app.py (PyQt6 Application)
    |
    ├─ calibration_system/
    │   ├─ calibration_app.py (Calibration UI)
    │   ├─ config.py (Configuration classes)
    │   ├─ camera_manager.py (Camera handling)
    │   ├─ homography.py (Coordinate transform)
    │   ├─ storage.py (JSON I/O)
    │   └─ widgets.py (Qt widgets)
    |
    ├─ models/
    │   └─ scripts/
    │       ├─ yolo_loader.py (YOLO model management)
    │       └─ config.py (Model configuration)
    |
    └─ src/
        ├─ multi_camera_detector.py (Detection engine)
        └─ heatmap_generator.py (Heatmap generation)

External Dependencies:
    ├─ OpenCV (cv2) - Computer vision
    ├─ NumPy - Array operations
    ├─ PyYAML/ultralytics - YOLO
    └─ PyQt6 - GUI framework
```

---

## Threading Model

### Calibration System

**Single-threaded** for simplicity and user interaction:
- User selects points sequentially
- Homography calculated on demand
- No concurrent processing needed

### Detection System

**Multi-threaded** for real-time processing:

```
Main Thread (GUI Event Loop)
    |
    ├─ Detection Thread 1 (Camera 1)
    ├─ Detection Thread 2 (Camera 2)
    ├─ Detection Thread 3 (Camera 3)
    └─ Detection Thread N (Camera N)
         |
         v
    Shared Detection Queue
         |
         v
    Storage Thread (JSON Write)
```

**Synchronization**:
- Thread-safe queue for detection data
- Lock for shared calibration config
- Event-based communication

---

## Data Structures

### CalibrationConfig

```python
@dataclass
class CameraConfig:
    camera_id: int
    name: str
    rtsp_url: str
    homography_matrix: ndarray      # 3x3
    camera_points: List[Point]      # Calibration points
    plan_points: List[Point]        # Floor plan points
    required_points: int            # Target points for calibration

@dataclass
class CalibrationConfig:
    store_name: str
    floor_plan_path: str
    cameras: List[CameraConfig]
    created_at: str
    updated_at: str
```

### Detection Record

```python
@dataclass
class DetectionPoint:
    timestamp: str                  # ISO format
    camera_id: int
    camera_name: str
    camera_point: Tuple[float, float]  # (x, y)
    plan_point: Tuple[float, float]    # (x, y)
    confidence: float               # 0.0-1.0
```

---

## Configuration Management

### Environment Variables

Managed via `.env` file:

```
RTSP_URL=rtsp://...                 # Camera connection
YOLO_MODEL=yolov8s.pt              # Model selection
CONFIDENCE_THRESHOLD=0.5           # Detection threshold
FRAME_WIDTH=640                    # Processing resolution
FRAME_HEIGHT=480
OUTPUT_DIR=./output                # Data storage
DEBUG_MODE=false                   # Logging level
```

### Calibration Files

Stored in `calibration_data/`:

```
calibration_mg_cite_olympique_20251217_114142.json
  └─ Full calibration data with all settings

homography_matrices_mg_cite_olympique.json
  └─ Only homography matrices (for production)
```

---

## Error Handling

### RTSP Connection Errors

```python
try:
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RTSPConnectionError(f"Failed to connect: {rtsp_url}")
except RTSPConnectionError as e:
    logger.error(f"[ERROR] {e}")
    # Retry logic or fallback
```

### YOLO Detection Errors

```python
try:
    detections = detector.detect_persons(frame)
except Exception as e:
    logger.error(f"[ERROR] Detection failed: {e}")
    return []  # Empty detections
```

### Homography Validation

```python
result = calculator.calculate_homography(camera_pts, plan_pts)
if result.reprojection_error > MAX_ERROR:
    logger.warning(f"High calibration error: {result.reprojection_error}")
    # Notify user to re-calibrate
```

### File I/O Errors

```python
try:
    with open(json_file) as f:
        data = json.load(f)
except FileNotFoundError:
    logger.error(f"[ERROR] Calibration file not found: {json_file}")
except json.JSONDecodeError:
    logger.error(f"[ERROR] Invalid JSON format: {json_file}")
```

---

## Performance Considerations

### Optimization Strategies

| Component | Optimization | Impact |
|-----------|-------------|--------|
| YOLO Detection | Smaller model (nano/small) | 50% faster, slightly less accurate |
| Frame Processing | Skip frames (process every Nth frame) | Faster, miss some detections |
| Heatmap Generation | Smaller histogram bins | Faster, less detailed |
| Visualization | Resize output image | Faster save time |

### Bottlenecks

1. **RTSP Network I/O** - Variable depending on camera and network
2. **YOLO Inference** - 30-50ms per frame (CPU), 5-10ms (GPU)
3. **Homography Transform** - Negligible (<1ms)
4. **JSON I/O** - Depends on detection count (typically <100ms)

### Scaling Recommendations

- For 1-2 cameras: CPU is sufficient
- For 3-4 cameras: Consider GPU acceleration
- For 5+ cameras: Distribute across multiple machines

---

## Future Architecture Improvements

### 1. Person Tracking

Add ByteTrack or DeepSORT to avoid counting same person multiple times:

```
YOLO Detection → Person Tracking → Unique IDs → Deduplication
```

### 2. Distributed Processing

Scale to multiple servers:

```
Camera 1-3 → Server A (Detection + Transform)
Camera 4-6 → Server B (Detection + Transform)
                |
                v
          Central Aggregator (Heatmap Generation)
```

### 3. Real-time Dashboard

WebSocket-based live heatmap updates:

```
Detection Engine → WebSocket Server → Web Browser (Live Heatmap)
```

### 4. Database Backend

Replace JSON with time-series database:

```
Detections → InfluxDB/TimescaleDB → Heatmap Generator
```

---

## Summary

The Traffic Heatmap system is designed with:

- **Modularity**: Clear separation between calibration, detection, and visualization
- **Scalability**: Multi-threading for concurrent camera processing
- **Robustness**: Error handling and validation at each stage
- **Configurability**: Environment and calibration-driven behavior
- **Extensibility**: Plugin architecture for future enhancements

This design allows for easy maintenance, testing, and future feature additions while maintaining real-time performance for typical retail deployments (4-8 cameras).
