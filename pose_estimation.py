import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation



# Global figure setup

fig= None
ax = None
line = None
points_scatter = None

def init_3d_plot():
    global fig,ax,line,points_scatter

    plt.ion()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111,projection='3d')

    #Initialize the camera trajectory line plot
    # Set the initial data to empty lists
    line_list = ax.plot([],[],[],'b-',label='Camera Trajectory')
    line = line_list[0]

    # Initialize a points 3d map scatter plot
    points_scatter = ax.scatter([],[],[],marker='.',c='r',s=1)

    ax.set_xlabel('X (Left / right)')
    ax.set_ylabel('Y (Depth)') # Note: Y is usually up/down in camera space, but let's use it for Z-depth here for map clarity
    ax.set_zlabel('Z (Up/Down)')

    # Set initial limits
    ax.set_xlim([-5,5])
    ax.set_ylim([-5,5])
    ax.set_zlim([-5,5])

    ax.legend()
    plt.title('SLAM Map and Camera Trajectory')
    plt.show(block=False)


def update_3d_plot(slam_map_instance):
    global ax,fig,points_scatter,line

    if not fig:
        return


    # Update trajectory
    # slam_map.trajectory is a list of [X,Y,Z] vectors
    if slam_map_instance.trajectory:
        path = np.array(slam_map_instance.trajectory)

        x_traj = path[:,0]
        y_traj = path[:,1]
        z_traj = path[:,2]

        line.set_data(x_traj,y_traj)
        line.set_3d_properties(z_traj)

        # -- Dynamic plot limits ---
        min_coords = path.min(axis=0)
        max_coords = path.max(axis=0)

        buffer = 1
        ax.set_xlim([min_coords[0] - buffer, max_coords[0] + buffer])
        ax.set_ylim([min_coords[1] - buffer,max_coords[1] + buffer])
        ax.set_zlim([min_coords[2] - buffer, max_coords[2] + buffer])

    # --- Update cloud point ----
    if slam_map_instance.point_cloud:
        # Concatenate all cloud chunks into one larg array
        all_points = np.hstack(slam_map_instance.point_cloud)

        points_scatter._offsets3d = (all_points[0],all_points[1],all_points[2])

    fig.canvas.draw_idle()
    fig.canvas.flush_events()





# --- 1. SLAM CONSTANTS ---
# Placeholder Camera Intrinsics Matrix (K)
# Assuming a standard 640x480 resolution.
# You would calibrate your camera for accurate values!
FOCAL_LENGTH = 713.8 # Generic focal length in pixels
PP_X = 319.5 # Principal point X
PP_Y = 239.5 # Principal point Y

K = np.array([
    [FOCAL_LENGTH, 0, PP_X],
    [0, FOCAL_LENGTH, PP_Y],
    [0, 0, 1]
], dtype=np.float64)


#  ------------ SLAM MAP MEMORY -----------
class SLAM_Map:
    def __init__(self):
        # 4x4 Transformation Matrix storing the camera's current global pose (Rotation and Translation)
        # Starts at the origin (Identity matrix)
        self.camera_pose = np.eye(4)

        # List to store 3D coordinats (X,Y,Z) of all map points
        self.point_cloud = []

        # List to store the camera's path (just the 3D translation vector [X, Y, Z])
        self.trajectory = []

        # Stores the frame indices (ID) of frames selected as keyframes
        self.keyframe_indices = []

        # Stores the actual grayscale images for the keyframes
        self.keyframe_images = []

        # Frame counter for tracking current index
        self.frame_count = 0

    def update_pose(self,R_delta,t_delta):
        # Create a 4x4 transformation matrix for the new movement
        T_delta = np.eye(4)
        T_delta[:3,:3] = R_delta
        T_delta[:3,3] = t_delta.flatten()

        # Integrate (multiply) the new movement into the current global pose
        # New_Pose = Old_Pose @ Delta_Movement (We multiply on the right for camera motion)
        self.camera_pose = self.camera_pose @ T_delta

        # Store the new global translation (trajectory)
        self.trajectory.append(self.camera_pose[:3,3].copy())

    def add_points(self,new_points_3d):
        # Convert the new points (3xN) into world coordinates (relative to the global pose)
        # R * p_c + t = p_w
        R = self.camera_pose[:3,:3]
        t = self.camera_pose[:3,3]

        points_w = R @ new_points_3d + t[:,None]

        self.point_cloud.append(points_w)





LK_PARAMS = dict(winSize=(21, 21),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Tracking variables
prev_keypoints = None
prev_frame = None
slam_map = SLAM_Map()




def pose_estimation():
    global prev_frame, prev_keypoints

    orb = cv2.ORB_create(nfeatures=2000)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('Error. Camera could not open.')
        return

    print("Running Project 7: 3D Pose Estimation & Triangulation. Press 'q' to exit.")

    init_3d_plot()

    # --- Find distinct landmarks / features in the frame / image

    def get_tracking_points(image_gray):
        points = cv2.goodFeaturesToTrack(
            image_gray,
            maxCorners=2000,
            qualityLevel=0.05,
            minDistance=7,
            blockSize=7
        )
        if points is None:
            return None

        return points.astype(np.float32).reshape(-1,1,2)



    def detect_loop(current_frame_gray,keyframe,keyframe_index):
        orb = cv2.ORB_create(nfeatures=500)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

        kp_curr,des_curr = orb.detectAndCompute(current_frame_gray,None)

        kp_hist,des_hist = orb.detectAndCompute(keyframe,None)

        if des_curr is None or des_hist is None:
            return None

        matches = matcher.match(des_curr,des_hist)

        if not isinstance(matches,list) or not matches:
            return None

        matches.sort(key=lambda x : x.distance)

        MIN_GOOD_MATCHES = 20

        if len(matches) > MIN_GOOD_MATCHES:

            src_pts = np.float32([kp_curr[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp_hist[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

            _,mask_F = cv2.findFundamentalMat(src_pts,dst_pts,cv2.FM_RANSAC,ransacReprojThreshold=3.0,confidence=0.99)

            inlier_count = np.sum(mask_F)

            if inlier_count >= 15:
                print(f'Loop detected! Frame {keyframe_index} matched with {inlier_count}')
                return keyframe_index

        return None



    while True:
        ret, frame = cap.read()
        if not ret: break

        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is None:
            initial_points = get_tracking_points(current_frame_gray)
            if initial_points is None:
                prev_frame = current_frame_gray
                continue


            prev_keypoints =  initial_points
            prev_frame = current_frame_gray
            continue


        # --- 2. Feature Tracking (Visual Odometry) ---

        # 2a. Guard Check: If features are somehow lost, re-initialize now
        if prev_keypoints is None or prev_keypoints.size < 8:
            print("Guard check: Features are lost. Re-initializing now...")
            new_points = get_tracking_points(current_frame_gray)

            if new_points is None:
                prev_frame = current_frame_gray
                continue

            prev_keypoints = new_points
            prev_frame = current_frame_gray
            continue # MUST continue if re-initialized

        # The lucas kanade finds quicky where each landmark moved from previous frame to next (current) one
        # 2b. Perform Tracking
        current_keypoints, status, err = cv2.calcOpticalFlowPyrLK(
            prev_frame, current_frame_gray, prev_keypoints, None, **LK_PARAMS
        )

        status_mask = status.ravel() == 1

        # 2c. Tracking Failure Check (Current Keypoints are None)
        if current_keypoints is None:
            print('Tracking failed entirely, current_keypoints is None. Re-initializing....')
            new_points = get_tracking_points(current_frame_gray)

            if new_points is None:
                prev_frame = current_frame_gray
                prev_keypoints = None
                continue

            prev_keypoints = new_points
            prev_frame = current_frame_gray
            continue # MUST continue if tracking failed

        # 2d. Filter and Check Point Count
        good_new = current_keypoints[status_mask]
        good_prev = prev_keypoints[status_mask]

        if len(good_new) < 8:
            print(f"Not enough points ({len(good_new)}) for reliable pose estimation. Re-detecting...")
            new_points = get_tracking_points(current_frame_gray)

            if new_points is None:
                prev_frame = current_frame_gray
                prev_keypoints = None
                continue

            prev_keypoints = new_points
            prev_frame = current_frame_gray
            continue # MUST continue if point count is too low



        # --- 3. Pose Calculation (The SLAM Back-End) ---

        # 3.1. Update Frame Count (for Loop Closure tracking)
        slam_map.frame_count += 1
        current_frame_index = slam_map.frame_count


        # 3.2. Check for Loop Closure periodically

        # Only check every 20 frames (and only if we have at least 5 keyframes to compare against)
        # You might need to adjust these parameters (20, 5) for your speed.
        if current_frame_index % 20 == 0 and len(slam_map.keyframe_images) >= 5:

        # We only match against the oldest frames (the start of the loop)
        # Checking against the first 5 keyframes is usually enough to detect the start of the loop
          for i in range(min(5, len(slam_map.keyframe_images))):
            matched_index = detect_loop(
            current_frame_gray,
            slam_map.keyframe_images[i],
            slam_map.keyframe_indices[i]
            )

            if matched_index is not None:
            # --- CORRECTION LOGIC ---
            # For simplicity, we only print the correction. A full correction requires optimization (PnP/BA).
                print(f"*** Loop Closure Successful! Correcting path from index {matched_index} onwards... ***")
            # In a real system, you would trigger a pose-graph optimization here.
            # Example: slam_map.correct_pose(matched_index, R, t)
                break


        # 3.3. Keyframe Creation (Add the current frame to historical memory)
        # Add a keyframe every 10 frames to build up historical memory
        if current_frame_index % 10 == 0:
         slam_map.keyframe_images.append(current_frame_gray.copy())
         slam_map.keyframe_indices.append(current_frame_index)

        # 3a. Find the Essential Matrix (E)
        # RANSAC is used here to robustly estimate E and filter out outliers.
        E, mask_E = cv2.findEssentialMat(
            good_new, good_prev, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )



        # 3b. Recover Pose (R, T) from the Essential Matrix
        # This function decomposes E into four possible (R, T) pairs and selects the one
        # that results in points having a positive depth (meaning they are in front of the camera).
        _, R, t, mask_pose = cv2.recoverPose(E, good_new, good_prev, K, mask=mask_E)

        scale = np.linalg.norm(t)

        if scale < 1e-6 or np.isnan(scale):
            print('Warning. Scale collapsed')
            t = t * 0.1

        slam_map.update_pose(R,t)

        # --- 3d world mapping with Triangulization
        mask_good = mask_E.ravel() == 1

        good_new_filtered = good_new[mask_good]
        good_prev_filtered = good_prev[mask_good]



        # Initialize points_3d and avg_z to safe default values
        points_3d = np.zeros((3, 0)) # Initialized as an empty (3, 0) array
        avg_z = 0

        # Triangulization

        if len(good_new_filtered) >= 1:
            # 1. Define camera projection matrices (p1 and p2)
           #p1 : The first camera (prev_frame) is placed at the origin [Identity rotation, Zero Translation]
            P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))

            # P2 : The second camera (current_frame) is defined by its calculated R and T
            P2 = K @ np.hstack((R,t))


            # 2. Triangulate the 3D points
            # cv2.triangulatePoints requires point coordinates in the format (2, N)
            # We use .T (transpose) to convert the shape from (N, 1, 2) to (2, N)
            points_4d = cv2.triangulatePoints(P1,P2,good_prev_filtered.reshape(-1,2).T, good_new_filtered.reshape(-1,2).T)

            # 3. Homogenius to cartesian conversion (4D TO 3D)
            # Triangulation returs points in 4D (X,Y,Z,W) , we convert to 3D (X,Y,Z)
            points_3d = points_4d[:3] / points_4d[3]

            slam_map.add_points(points_3d)

            if points_3d.shape[1] > 0:
               positive_depth_mask = points_3d[2] > 0
               valid_points = points_3d[:,positive_depth_mask]

               if valid_points.shape[1] > 0:
                  avg_z = np.mean(valid_points[2])
               else:
                  avg_z = 0




        # --- 4. Visualization and Update ---

        img = frame.copy()

        cv2.putText(img,f'Avg Z depth (MAP) : {avg_z:.2f}',(10,130),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)


        mask = np.zeros_like(frame)

        # Only draw RANSAC-inlier points
        for i, (new, old) in enumerate(zip(good_new_filtered, good_prev_filtered)):
            a, b = map(int, new.ravel())
            c, d = map(int, old.ravel())

            # Draw the tracking line (on mask)
            mask = cv2.line(mask, (a, b), (c, d), (255, 0, 0), 2) # Blue line
            # Draw the new point (on frame)
            frame = cv2.circle(frame, (a, b), 3, (0, 0, 255), -1) # Red circle


        img = cv2.add(frame,mask)


        cv2.putText(img,f'Avg Z depth (MAP) : {avg_z:.2f}',(10,130),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

        # --- 5. Display Pose Data ---
        # Display the calculated rotation and translation (visual odometry data)
        # R is a 3x3 matrix, t is a 3x1 vector
        cv2.putText(img, f"R:\n{R[0,:]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"T: [{t[0,0]:.2f}, {t[1,0]:.2f}, {t[2,0]:.2f}]", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display
        cv2.imshow('Project 7: 3D Pose Estimation (SLAM Back-End)', img)

        update_3d_plot(slam_map)

        # Update for next iteration (Use RANSAC inliers for better robustness)
        prev_frame = current_frame_gray.copy()
        prev_keypoints = good_new_filtered.reshape(-1,1,2)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    pose_estimation()
