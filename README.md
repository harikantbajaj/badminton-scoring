# 🏸 Badminton Scoring System

An advanced computer vision-based badminton scoring system that automatically detects players, shuttlecock, and court boundaries to score points in real-time from video footage.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Models and Datasets](#models-and-datasets)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Real-time Player Detection**: Uses YOLOv8 to detect and track badminton players
- **Shuttlecock Tracking**: Advanced shuttlecock detection with trajectory visualization
- **Court Boundary Detection**: Automatic court keypoint detection and boundary drawing
- **Automated Scoring**: Intelligent scoring system based on shuttle landing positions
- **Video Annotation**: Outputs annotated videos with scores, trajectories, and court overlays
- **GPU Support**: Optimized for CUDA acceleration when available
- **Multiple Detection Modes**: Separate scripts for different detection tasks

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for better performance)

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd badminton-scoring
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download required models (see [Models and Datasets](#models-and-datasets) section)

## 📖 Usage

### Basic Player and Shuttle Detection

Run the main detection script:

```bash
python main.py
```

This will process `videos/00001.mp4` and display real-time detection of players and shuttlecock.

### Court Boundary Detection

Detect court keypoints and draw boundaries:

```bash
python detect_court.py
```

Processes `videos/00353.mp4` and shows court detection with player zones.

### Shuttlecock Tracking Only

Track shuttlecock trajectory:

```bash
python detect_objects.py
```

Processes `videos/00777.mp4` with advanced shuttle tracking and trajectory visualization.

### Full Scoring System

Run the complete automated scoring system:

```bash
python both.py
```

Combines court detection, shuttle tracking, and scoring logic. Outputs annotated video to `output/annotated_00007.mp4`.

### Custom Video Processing

To process your own videos, modify the `video_path` variable in the respective scripts:

```python
video_path = "path/to/your/video.mp4"
```

## 📁 Project Structure

```
badminton-scoring/
├── main.py                 # Basic player/shuttle detection
├── detect_court.py         # Court boundary detection
├── detect_objects.py       # Shuttlecock tracking
├── both.py                 # Full scoring system
├── download_dataset.py     # Dataset download script
├── requirements.txt        # Python dependencies
├── TODO.md                 # Project tasks and progress
├── models/                 # YOLO datasets
│   ├── Badminton Court Keypoint Dataset.v5i.yolov8/
│   └── Shuttlecock-detection-3/
├── utils/
│   └── detection.py        # BadmintonDetector class
├── videos/                 # Input video files
├── output/                 # Processed video outputs
└── runs/                   # Training results and weights
    ├── detect/             # Object detection training runs
    └── pose/               # Pose estimation training runs
```

## 📦 Dependencies

- `ultralytics>=8.0.0` - YOLOv8 implementation
- `opencv-python>=4.5.0` - Computer vision library
- `numpy>=1.21.0` - Numerical computing
- `torch>=1.10.0` - PyTorch deep learning framework
- `torchvision>=0.11.0` - PyTorch vision utilities

## 🤖 Models and Datasets

### Pre-trained Models

The project uses several YOLOv8 models:

- **Player Detection**: `yolov8n.pt` (COCO pre-trained)
- **Shuttlecock Detection**: Custom trained model in `runs/detect/train8/weights/best.pt`
- **Court Keypoints**: Pose estimation model in `runs/pose/train19/weights/best.pt`

### Datasets

- **Shuttlecock Dataset**: Downloaded from Roboflow (Shuttlecock-detection-3)
- **Court Dataset**: Badminton Court Keypoint Dataset from Roboflow

### Training Your Own Models

To train custom models:

```python
from ultralytics import YOLO

# Load and train model
model = YOLO('yolov8n.pt')
model.train(data='path/to/dataset.yaml', epochs=100)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Test on multiple video formats
- Update README for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 Future Enhancements

- [ ] Multi-camera support for 3D tracking
- [ ] Real-time webcam processing
- [ ] Advanced scoring rules implementation
- [ ] Player pose analysis for technique evaluation
- [ ] Web interface for video upload and processing
- [ ] Mobile app integration

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**: Ensure model paths in scripts match your trained models
2. **Video not loading**: Check video file format and path
3. **CUDA errors**: Install CUDA toolkit or set `device='cpu'`
4. **Low detection accuracy**: Adjust confidence thresholds in scripts

### Performance Tips

- Use GPU for faster processing
- Lower confidence thresholds for more detections
- Process shorter video clips for testing
- Use higher resolution videos for better accuracy

---

**Note**: This system is designed for badminton video analysis and scoring. Ensure videos are well-lit and captured from appropriate angles for best results.
