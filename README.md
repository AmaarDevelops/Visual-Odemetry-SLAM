# 🧭 VINS-SLAM-Monocular

A complete real-time Monocular Visual-Inertial SLAM (VINS-SLAM) system that achieves metric-scale 3D pose estimation and mapping using a single camera and simulated Inertial Measurement Unit (IMU) data.

This project goes beyond standard Visual Odometry (VO) by implementing sensor fusion for **true metric scaling** and a back-end structure for **Loop Closure** and **Relocalization**.

## ✨ Key Features & Technical Achievements

| Feature | Description | Achievement |
| :--- | :--- | :--- |
| **Monocular VO** | Tracks features (Lucas-Kanade) and computes the relative camera pose (R, t) between frames. | Standard Front-End |
| **VINS Integration** | Fuses scale-less visual data with integrated inertial data. | **Core Achievement:** Solved the scale ambiguity problem of monocular VO. |
| **Metric Scale Fix** | Calculates the true distance traveled (in **meters**) from the IMU and applies the scale factor to the visual translation vector (t). | **Metric Accuracy:** Trajectory is defined in real-world units. |
| **Loop Closure** | Detects when the camera revisits a known location using feature matching (ORB). | **Error Correction:** Implemented a heuristic pose warp to correct accumulated drift. |
| **Relocalization** | Ability to re-acquire the camera's pose by matching against historical keyframes when visual tracking is lost. | **Robustness:** System can recover from tracking failure. |
| **3D Mapping** | Uses Triangulation to build a dense 3D point cloud of the environment in world coordinates. | Real-time map generation. |

## ⚙️ How It Works: The Pipeline

The system operates in a continuous, three-part loop: 

1.  **Front-End (Visual Odometry):** Features are tracked across two frames. The Essential Matrix (E) is computed, and the relative rotation and translation direction ($\mathbf{R}, \mathbf{t}_{\text{direction}}$) are recovered.
2.  **Sensor Fusion (VINS):** The camera's global rotation ($\mathbf{R}_{\text{global}}$) is used to transform the noisy IMU acceleration into the world frame. This acceleration is integrated twice to find the ground truth metric distance ($\Delta t_{\text{IMU}}$). The ratio $\frac{\Vert \Delta t_{\text{IMU}} \Vert}{\Vert t_{\text{direction}} \Vert}$ provides the necessary metric scale factor.
3.  **Back-End (SLAM):** The scaled pose is integrated into the global map. Keyframes are stored. Periodically, loop closure checks run to detect revisited areas and correct the overall map drift.

## 🚀 Setup and Run

### Prerequisites

* Python 3.x
* OpenCV (`cv2`)
* NumPy
* Matplotlib (for 3D visualization)

### Installation

``bash

# Clone the repository
git clone [https://github.com/AmaarDevelops/VINS-SLAM-Monocular.git]
cd VINS-SLAM-Monocular

# Install required Python packages
pip install opencv-python numpy matplotlib
Execution
The script utilizes your webcam (cv2.VideoCapture(0)) for real-time visual input.

Bash

python your_main_file_name.py
Output 1: A window showing the camera feed, feature tracking, and VINS/pose data.

Output 2: A Matplotlib 3D window displaying the camera's metric trajectory (blue line) and the generated 3D point cloud (red dots).

To exit the application, press 'q' while the video window is active.
