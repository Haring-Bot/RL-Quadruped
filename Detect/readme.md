# 🤖 Robot Vision System - Object Detection & 3D Localization

A complete computer vision system for detecting objects (cube, sphere, cylinder) in 3D space using PyBullet simulation, YOLOv8, and depth estimation. Perfect for robotics projects, pick-and-place tasks, and autonomous navigation.

## 🎯 What Does This Do?

This system:
- **Detects** objects in a simulated camera view using YOLOv8
- **Measures** distance to each object using depth buffers
- **Calculates** 3D position (X, Y, Z coordinates) in world space
- **Provides** accurate target coordinates for robot path planning

Perfect for robotics students and researchers working on object manipulation and navigation tasks!

---

## 📦 Project Structure

```
📦Detect
 ┣ 📂config
 ┃ ┗ 📜config.yaml              # All system parameters
 ┣ 📂scripts
 ┃ ┣ 📜test_detection.py        # Test detection accuracy
 ┃ ┗ 📜test_path.py             # Test path planning interface
 ┣ 📂src
 ┃ ┣ 📂detection
 ┃ ┃ ┗ 📜detector.py            # YOLO object detector
 ┃ ┣ 📂perception
 ┃ ┃ ┗ 📜depth_estimator.py     # Distance & position calculation
 ┃ ┣ 📂simulation
 ┃ ┃ ┣ 📜camera.py              # PyBullet camera interface
 ┃ ┃ ┗ 📜environment.py         # Simulation environment
 ┃ ┣ 📂tracking
 ┃ ┃ ┗ 📜object_tracker.py      # Object tracking (optional)
 ┃ ┗ 📜vision_system.py         # Main vision system
 ┣ 📂weights
 ┃ ┗ 📜best.pt                  # Trained YOLOv8 model (2000 images)
 ┗ 📜requirements.txt            # Python dependencies
```

---

## 💻 Windows Setup Guide (Conda Installation)

### Step 1: Install Miniconda/Anaconda

**Option A: Miniconda (Recommended - Lighter)**
1. Download Miniconda for Windows from: https://docs.conda.io/en/latest/miniconda.html
2. Choose "Miniconda3 Windows 64-bit" installer
3. Run the installer (.exe file)
4. **Important:** Check "Add Anaconda to my PATH environment variable" during installation
5. Complete the installation

**Option B: Anaconda (Full Package)**
1. Download from: https://www.anaconda.com/download
2. Run the installer
3. Follow the installation wizard

### Step 2: Verify Installation

Open **Command Prompt** or **Anaconda Prompt** and test:
```bash
conda --version
# Should show: conda 23.x.x or similar
```

If you get an error:
1. Close and reopen Command Prompt
2. Or search for "Anaconda Prompt" in Windows Start Menu and use that instead

### Step 3: Update Conda (Optional but Recommended)
```bash
conda update conda
```

### Step 4: Common Windows Issues

**Issue: "conda is not recognized"**
- Use **Anaconda Prompt** instead of regular Command Prompt
- Or add conda to PATH:
  1. Search "Environment Variables" in Windows
  2. Edit "Path" variable
  3. Add: `C:\Users\YourUsername\miniconda3\Scripts`
  4. Add: `C:\Users\YourUsername\miniconda3\Library\bin`

**Issue: Permission Denied**
- Run Command Prompt or Anaconda Prompt as Administrator
- Right-click → "Run as administrator"

**Issue: Slow conda commands**
```bash
# Use faster libmamba solver
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

---

## 🚀 Quick Start

### 1. Installation

**For Windows Users:**
```bash
# Open Anaconda Prompt (NOT regular Command Prompt)
# Navigate to your project directory
cd C:\path\to\your\folder

# Clone the repository
git clone <your-repo-url>
cd Detect

# Create conda virtual environment
conda create -n robot_vision python=3.11
# Press 'y' when asked to proceed

# Activate environment
conda activate robot_vision
# You should see (robot_vision) in your prompt

# Install dependencies
pip install -r requirements.txt
# This may take 5-10 minutes - be patient!
```

**For Linux/Mac Users:**
```bash
# Open Terminal
cd /path/to/your/folder

# Clone the repository
git clone <your-repo-url>
cd Detect

# Create conda virtual environment
conda create -n robot_vision python=3.11
conda activate robot_vision

# Install dependencies
pip install -r requirements.txt
```

**Note:** This project was developed using conda for environment management. Python 3.11 is recommended.

### 2. Test the System

**Test Detection Accuracy:**
```bash
python scripts/test_detection.py --scenes 5 --objects 3
```

**Test Path Planning Interface:**
```bash
python scripts/test_path.py
```

### 3. Use in Your Project

```python
import sys
sys.path.append('src')

from simulation.environment import RobotEnvironment
from simulation.camera import Camera
from vision_system import VisionSystem

# Setup
env = RobotEnvironment(gui=True)
env.spawn_random_scene(num_each_type=2)

camera = Camera()
vision = VisionSystem(camera, model_path="weights/best.pt")

# Run detection
result = vision.detect_and_measure()

# Get target coordinates for path planning
for detection in result['detections']:
    target_x = detection['position'][0]  # X coordinate
    target_y = detection['position'][1]  # Y coordinate
    object_type = detection['class_name']
    
    print(f"{object_type} at [{target_x:.3f}, {target_y:.3f}]")
```

---

## 🎓 How It Works

### 1. **Object Detection** (YOLOv8)
- Trained on 2000 synthetic images
- Detects: `cube`, `sphere`, `cylinder`
- Returns bounding boxes and confidence scores

### 2. **Distance Estimation** (Depth Buffer)
- Uses PyBullet's depth buffer
- Converts depth to real-world meters
- Validates measurements for accuracy

### 3. **3D Position Calculation**
- Transforms pixel coordinates to 3D world coordinates
- Uses camera intrinsics (focal length, principal point)
- Accounts for camera pose and orientation

### 4. **Coordinate System**
```
World Frame:
  X-axis: Left/Right
  Y-axis: Forward/Back  
  Z-axis: Up/Down

Camera Frame:
  Position: [0, -1.8, 0.6] (behind workspace, elevated)
  Looking at: [0, 0, 0.4] (table height)
```

---

## 📊 Performance Metrics

**Trained Model Performance:**
- Training Dataset: 2000 images
- Distance Range: 0.8m - 1.8m
- Angle Range: -15° to +15°

**Expected Accuracy:**
- F1 Score Target: ≥ 80%
- Distance Error: < 15cm
- Position Error: < 20cm

---

## ⚙️ Configuration

All parameters are in `config/config.yaml`:

### Object Definitions
```yaml
objects:
  sphere:
    diameter: 0.15
    color: [0.0, 0.0, 1.0, 1.0]  # Blue
    class_id: 1
  cube:
    size: [0.20, 0.20, 0.20]
    color: [1.0, 0.0, 0.0, 1.0]  # Red
    class_id: 0
  cylinder:
    height: 0.16
    radius: 0.06
    color: [0.0, 1.0, 0.0, 1.0]  # Green
    class_id: 2
```

### Camera Settings
```yaml
camera:
  width: 640
  height: 480
  fov: 60
  position: [0, -1.8, 0.6]
  target: [0, 0, 0.4]
```

### Detection Parameters
```yaml
detection:
  model: "yolov8n.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.45
```

---

## 🧪 Testing

### Run Full Test Suite
```bash
# Test with 10 scenes, 5 objects each
python scripts/test_detection.py --scenes 10 --objects 5

# Test without visualization (faster)
python scripts/test_detection.py --scenes 20 --objects 3 --no-viz
```

### Expected Test Output
```
TEST RESULTS
======================================================================
Overall Performance:
  Total objects: 50
  Detected: 45
  Missed: 5
  False positives: 2

Metrics:
  Precision: 95.74%
  Recall: 90.00%
  F1 Score: 92.78%

Distance Estimation:
  Mean error: 0.089m
  Std dev: 0.067m
  Max error: 0.234m
```

---

## 💡 Quick Tips for Windows Users

### Running Scripts
Always use **Anaconda Prompt** and make sure environment is activated:
```bash
# Check if environment is active (you should see it in prompt)
# Correct: (robot_vision) C:\Users\YourName\Detect>
# Wrong:   C:\Users\YourName\Detect>

# If not active, activate it:
conda activate robot_vision

# Then run scripts:
python scripts\test_detection.py
python scripts\test_path.py
```

### File Paths on Windows
Use backslashes `\` or forward slashes `/`:
```bash
# Both work on Windows:
python scripts\test_detection.py
python scripts/test_detection.py
```

### Common Shortcuts
- `Ctrl + C`: Stop running script
- `cls`: Clear terminal screen
- `dir`: List files (instead of `ls`)
- `cd ..`: Go up one directory

### GPU Support (Optional - for faster training)
If you have NVIDIA GPU:
```bash
# Install CUDA-enabled PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🎯 Use Cases

### 1. **Pick and Place Robot**
```python
# Detect objects
result = vision.detect_and_measure()

# Plan path to each object
for obj in result['detections']:
    target = [obj['position'][0], obj['position'][1]]
    robot.move_to(target)
    robot.pick_up(obj['class_name'])
```

### 2. **Object Sorting**
```python
# Detect and sort by type
cubes = [d for d in detections if d['class_name'] == 'cube']
spheres = [d for d in detections if d['class_name'] == 'sphere']

# Process each category
for cube in cubes:
    robot.move_to_bin('cube_bin', cube['position'])
```

### 3. **Navigation Planning**
```python
# Detect obstacles
obstacles = vision.detect_and_measure()['detections']

# Get obstacle positions for path planner
obstacle_positions = [d['position'][:2] for d in obstacles]
path = path_planner.plan(start, goal, obstacles=obstacle_positions)
```

---

## 📚 Dependencies

**Environment Management:**
- **Conda**: This project uses conda for virtual environment management
- **Python**: 3.11 (recommended)

Main libraries:
- **PyBullet** (≥3.2.5): Physics simulation
- **YOLOv8/Ultralytics** (≥8.0.0): Object detection
- **OpenCV** (≥4.8.0): Image processing
- **NumPy** (≥1.24.0): Numerical computing
- **PyTorch** (≥2.0.0): Deep learning backend
- **PyYAML** (≥6.0): Configuration files

See `requirements.txt` for complete list.

**Managing your environment:**
```bash
# Activate environment
conda activate robot_vision

# Deactivate when done
conda deactivate

# Remove environment (if needed)
conda env remove -n robot_vision
```



## 👨‍💻 Author

Created for robotics research and education. Trained on 2000 synthetic images generated in PyBullet simulation.
By THU HTOO ZAW, THIRI TOE TOE ZIN

---



