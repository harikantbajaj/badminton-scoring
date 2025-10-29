import cv2
from ultralytics import YOLO
from collections import deque
import torch
import numpy as np
import os

# ===========================
# 🔹 GPU Check
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🟢 Using device: {device}")

# ===========================
# 🔹 Load YOLO Models
# ===========================
court_model_path = r"C:\badm\runs\pose\train19\weights\best.pt"
shuttle_model_path = r"C:\badm\runs\detect\train8\weights\best.pt"

if not os.path.exists(court_model_path):
    raise FileNotFoundError(f"❌ Court model not found: {court_model_path}")
if not os.path.exists(shuttle_model_path):
    raise FileNotFoundError(f"❌ Shuttle model not found: {shuttle_model_path}")

court_model = YOLO(court_model_path)
shuttle_model = YOLO(shuttle_model_path)
print("✅ Models Loaded Successfully")

# ===========================
# 🔹 Load Video
# ===========================
video_path = r"C:\badm\badminton-scoring\videos\00014.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"❌ Cannot open video: {video_path}")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📹 Video FPS: {fps}, Total Frames: {total_frames}, Resolution: {width}x{height}")

# ===========================
# 🔹 Video Writer Setup
# ===========================
output_dir = r"C:\badm\badminton-scoring\output"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "annotated_00007.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try 'avc1' or 'H264'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

if not out.isOpened():
    print("⚠️ Warning: Could not open video writer. Trying alternative codec...")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = os.path.join(output_dir, "annotated_00794.avi")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"💾 Saving output to: {output_path}")

# ===========================
# 🔹 Scoring Variables
# ===========================
player1_score = 0
player2_score = 0
shuttle_traj = deque(maxlen=50)
velocity_history = deque(maxlen=20)
frame_count = 0
rally_active = False
last_scoring_frame = -100
scoring_cooldown = 45

# Shuttle tracking state
shuttle_state = {
    'last_pos': None,
    'last_side': None,
    'crossed_net': False,
    'ground_frames': 0,
    'missing_frames': 0,
    'last_detected_frame': 0,
    'scored_this_rally': False,
    'low_position_frames': 0,
    'landing_position': None,
    'landing_side': None
}

shift_ratio = 0.33

# ===========================
# 🔹 Helper Functions
# ===========================
def get_ordered_corners(points):
    """Compute convex hull and return ordered corners TL, TR, BR, BL"""
    hull = cv2.convexHull(points.astype(np.float32))
    hull = hull[:, 0, :]
    
    s = hull.sum(axis=1)
    diff = np.diff(hull, axis=1)
    
    TL = hull[np.argmin(s)]
    BR = hull[np.argmax(s)]
    TR = hull[np.argmin(diff)]
    BL = hull[np.argmax(diff)]
    
    return np.array([TL, TR, BR, BL], dtype=np.int32)

def draw_dashed_line(img, pt1, pt2, color=(0, 0, 255), thickness=3, dash_length=10):
    """Draw dashed line between two points"""
    pt1 = np.array(pt1, dtype=int)
    pt2 = np.array(pt2, dtype=int)
    line_len = np.linalg.norm(pt2 - pt1)
    dashes = int(line_len // dash_length)
    for i in range(dashes):
        start = pt1 + (pt2 - pt1) * (i / dashes)
        end = pt1 + (pt2 - pt1) * ((i + 0.5) / dashes)
        cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness)

def point_in_polygon(point, polygon):
    """Check if point is inside polygon"""
    return cv2.pointPolygonTest(polygon, tuple(point), False) >= 0

def which_side(point, net_y):
    """Determine which side of court the point is on"""
    if point[1] < net_y:
        return 'p2'
    else:
        return 'p1'

def calculate_velocity(traj, window=3):
    """Calculate shuttle velocity from trajectory"""
    if len(traj) < window + 1:
        return 0, 0
    
    recent = list(traj)[-window-1:]
    dx = recent[-1][0] - recent[0][0]
    dy = recent[-1][1] - recent[0][1]
    
    return dx / window, dy / window

def is_shuttle_grounded(shuttle_pos, ground_threshold, velocity, traj_history):
    """
    Detect if shuttle is on the ground with IMPROVED criteria
    """
    if len(traj_history) < 2:
        return False, 0
    
    y_pos = shuttle_pos[1]
    recent_positions = list(traj_history)[-5:]
    y_positions = [p[1] for p in recent_positions]
    
    # Criterion 1: Position in ground zone (MORE LENIENT)
    in_ground_zone = y_pos >= ground_threshold - 50
    
    if not in_ground_zone:
        return False, 0
    
    # Criterion 2: Calculate speed
    vx, vy = velocity
    speed = np.sqrt(vx**2 + vy**2)
    
    # Criterion 3: Position stability
    y_variance = np.var(y_positions) if len(y_positions) >= 2 else 1000
    
    # Criterion 4: Not rising rapidly
    is_not_rising = vy >= -5
    
    # IMPROVED Scoring system
    confidence = 0
    
    if in_ground_zone:
        confidence += 35
    
    if speed < 8.0:
        confidence += 30
    elif speed < 15.0:
        confidence += 15
    
    if y_variance < 300:
        confidence += 20
    elif y_variance < 500:
        confidence += 10
    
    if is_not_rising:
        confidence += 15
    
    is_grounded = confidence >= 50
    
    return is_grounded, confidence

def check_out_of_bounds(point, court_polygon, margin=30):
    """Check if shuttle is outside court"""
    result = cv2.pointPolygonTest(court_polygon, tuple(point), True)
    return result < -margin

def award_point(landing_side, landing_pos, p1_court, p2_court, frame_count):
    """Award point based on landing position"""
    global player1_score, player2_score, scoring_event, landing_marker, landing_marker_timer
    
    # Check bounds
    if landing_side == 'p1':
        in_bounds = point_in_polygon(landing_pos, p1_court)
        out = check_out_of_bounds(landing_pos, p1_court)
    else:
        in_bounds = point_in_polygon(landing_pos, p2_court)
        out = check_out_of_bounds(landing_pos, p2_court)
    
    # Award point
    if in_bounds and not out:
        if landing_side == 'p1':
            player2_score += 1
            scoring_event = "PLAYER 2 SCORES!"
            print(f"\n🎯 Frame {frame_count}: PLAYER 2 SCORES! (In P1 court)")
        else:
            player1_score += 1
            scoring_event = "PLAYER 1 SCORES!"
            print(f"\n🎯 Frame {frame_count}: PLAYER 1 SCORES! (In P2 court)")
        
        landing_marker = landing_pos
        landing_marker_timer = 90
        return True
        
    elif out:
        if landing_side == 'p1':
            player1_score += 0
            scoring_event = "OUT! PLAYER 1 SCORES!"
            print(f"\n🎯 Frame {frame_count}: OUT! PLAYER 1 SCORES!")
        else:
            player2_score += 0
            scoring_event = "OUT! PLAYER 2 SCORES!"
            print(f"\n🎯 Frame {frame_count}: OUT! PLAYER 2 SCORES!")
        
        landing_marker = landing_pos
        landing_marker_timer = 90
        return True
    
    return False

# ===========================
# 🔹 Main Processing Loop
# ===========================
court_corners = None
net_line_y = None
p1_court = None
p2_court = None

scoring_event = None
scoring_event_timer = 0
landing_marker = None
landing_marker_timer = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n🎬 VIDEO ENDED - Checking final state...")
        
        if shuttle_state['last_pos'] is not None:
            print(f"⚠️ Shuttle state at video end:")
            print(f"  Last position: {shuttle_state['last_pos']}")
            print(f"  Last side: {shuttle_state['last_side']}")
            print(f"  Missing frames: {shuttle_state['missing_frames']}")
            print(f"  Ground frames: {shuttle_state['ground_frames']}")
            print(f"  Low position frames: {shuttle_state['low_position_frames']}")
            print(f"  Scored this rally: {shuttle_state['scored_this_rally']}")
            
            should_score = (
                (rally_active or shuttle_state['low_position_frames'] > 0) and
                not shuttle_state['scored_this_rally'] and
                p1_court is not None and 
                p2_court is not None and
                shuttle_state['last_side'] is not None
            )
            
            if should_score:
                landing_pos = shuttle_state['landing_position'] if shuttle_state['landing_position'] is not None else shuttle_state['last_pos']
                landing_side = shuttle_state['landing_side'] if shuttle_state['landing_side'] is not None else shuttle_state['last_side']
                
                print(f"\n🎯 AWARDING FINAL POINT...")
                awarded = award_point(landing_side, landing_pos, p1_court, p2_court, frame_count)
                
                if awarded:
                    print(f"✅ Final point awarded successfully!")
                else:
                    print(f"⚠️ Final point could not be awarded (invalid position)")
        
        break
    
    frame_count += 1
    annotated_frame = frame.copy()
    
    # Show progress
    if frame_count % 100 == 0:
        progress = (frame_count / total_frames) * 100
        print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)...")
    
    # ===========================
    # 🔹 Court Detection
    # ===========================
    court_results = court_model.predict(source=frame, conf=0.35, verbose=False)
    
    for result in court_results:
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            continue
        
        keypoints = result.keypoints.xy[0].cpu().numpy()
        if hasattr(result.keypoints, 'conf'):
            confs = result.keypoints.conf[0].cpu().numpy()
            keypoints = keypoints[confs > 0.3]
        
        if len(keypoints) < 4:
            continue
        
        TL, TR, BR, BL = get_ordered_corners(keypoints)
        court_corners = np.array([TL, TR, BR, BL], np.int32)
        
        # Draw court
        cv2.polylines(annotated_frame, [court_corners], isClosed=True, 
                     color=(0, 255, 0), thickness=3)
        
        # Net line
        net_line_y = int(BL[1] * shift_ratio + TL[1] * (1 - shift_ratio))
        net_left = (min(TL[0], TR[0], BL[0], BR[0]), net_line_y)
        net_right = (max(TL[0], TR[0], BL[0], BR[0]), net_line_y)
        
        draw_dashed_line(annotated_frame, net_left, net_right, 
                        color=(0, 0, 255), thickness=3, dash_length=15)
        
        # Define courts
        p1_court = np.array([BL, BR, [BR[0], net_line_y], [BL[0], net_line_y]], np.int32)
        p2_court = np.array([TL, TR, [TR[0], net_line_y], [TL[0], net_line_y]], np.int32)
        
        # Labels
        p1_text_pos = (BL[0] + 30, (BL[1] + net_line_y) // 2)
        p2_text_pos = (TL[0] + 30, (TL[1] + net_line_y) // 2)
        cv2.putText(annotated_frame, "PLAYER 1", p1_text_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated_frame, "PLAYER 2", p2_text_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # ===========================
    # 🔹 Shuttle Detection
    # ===========================
    shuttle_results = shuttle_model(frame, conf=0.40, device=device, verbose=False)
    
    shuttle_detected = False
    current_shuttle = None
    
    if shuttle_results and shuttle_results[0].boxes is not None:
        for box in shuttle_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            current_shuttle = (cx, cy)
            shuttle_detected = True
            
            conf = float(box.conf[0])
            cv2.circle(annotated_frame, (cx, cy), 6, (0, 255, 0), -1)
            cv2.circle(annotated_frame, (cx, cy), 10, (255, 255, 0), 2)
            cv2.putText(annotated_frame, f"{conf:.2f}", (cx + 12, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            break
    
    # Track missing frames
    if not shuttle_detected:
        shuttle_state['missing_frames'] += 1
    else:
        shuttle_state['missing_frames'] = 0
        shuttle_state['last_detected_frame'] = frame_count
    
    # ===========================
    # 🔹 Scoring Logic
    # ===========================
    if current_shuttle and court_corners is not None and net_line_y is not None:
        shuttle_traj.append(current_shuttle)
        
        # Calculate velocity
        vx, vy = calculate_velocity(shuttle_traj, window=3)
        velocity_history.append((vx, vy))
        
        current_side = which_side(current_shuttle, net_line_y)
        
        # Net crossing detection
        if shuttle_state['last_side'] is not None and shuttle_state['last_side'] != current_side:
            if not rally_active:
                rally_active = True
                shuttle_state['crossed_net'] = True
                shuttle_state['scored_this_rally'] = False
                shuttle_state['ground_frames'] = 0
                shuttle_state['low_position_frames'] = 0
                print(f"\n🎾 Frame {frame_count}: NET CROSSED! Rally active on {current_side} side")
        
        # Ground thresholds
        if current_side == 'p1':
            ground_threshold = BL[1] - 100
        else:
            ground_threshold = TL[1] + 100
        
        # Track if shuttle is in low position
        if current_shuttle[1] >= ground_threshold - 50:
            shuttle_state['low_position_frames'] += 1
            shuttle_state['landing_position'] = current_shuttle
            shuttle_state['landing_side'] = current_side
        else:
            if current_shuttle[1] < ground_threshold - 100:
                shuttle_state['low_position_frames'] = 0
        
        # Ground detection
        if (rally_active or shuttle_state['low_position_frames'] > 0) and not shuttle_state['scored_this_rally']:
            is_grounded, confidence = is_shuttle_grounded(
                current_shuttle, ground_threshold, (vx, vy), shuttle_traj
            )
            
            if is_grounded:
                shuttle_state['ground_frames'] += 1
                shuttle_state['landing_position'] = current_shuttle
                shuttle_state['landing_side'] = current_side
                print(f"🔍 Frame {frame_count}: Ground detected! Frames: {shuttle_state['ground_frames']}, Confidence: {confidence}")
            else:
                if shuttle_state['ground_frames'] > 0 and current_shuttle[1] < ground_threshold - 120:
                    shuttle_state['ground_frames'] = 0
            
            if shuttle_state['ground_frames'] >= 1:
                can_score = (frame_count - last_scoring_frame) > scoring_cooldown
                
                if can_score:
                    landing_pos = shuttle_state['landing_position']
                    landing_side = shuttle_state['landing_side']
                    
                    awarded = award_point(landing_side, landing_pos, p1_court, p2_court, frame_count)
                    
                    if awarded:
                        last_scoring_frame = frame_count
                        scoring_event_timer = 80
                        shuttle_state['scored_this_rally'] = True
                        shuttle_state['ground_frames'] = 0
                        rally_active = False
        
        shuttle_state['last_side'] = current_side
        shuttle_state['last_pos'] = current_shuttle
    
    # Draw trajectory
    for i in range(1, len(shuttle_traj)):
        cv2.line(annotated_frame, shuttle_traj[i-1], shuttle_traj[i], (0, 255, 255), 2)
    
    # Draw velocity arrow
    if len(velocity_history) > 0 and current_shuttle:
        vx, vy = velocity_history[-1]
        end_point = (int(current_shuttle[0] + vx * 4), int(current_shuttle[1] + vy * 4))
        cv2.arrowedLine(annotated_frame, current_shuttle, end_point, (255, 0, 255), 2)
    
    # Draw landing marker
    if landing_marker and landing_marker_timer > 0:
        if "OUT" in str(scoring_event):
            cv2.circle(annotated_frame, tuple(map(int, landing_marker)), 20, (0, 0, 255), 3)
            cv2.line(annotated_frame, 
                    (int(landing_marker[0])-15, int(landing_marker[1])-15),
                    (int(landing_marker[0])+15, int(landing_marker[1])+15), (0, 0, 255), 3)
            cv2.line(annotated_frame,
                    (int(landing_marker[0])-15, int(landing_marker[1])+15),
                    (int(landing_marker[0])+15, int(landing_marker[1])-15), (0, 0, 255), 3)
        else:
            cv2.circle(annotated_frame, tuple(map(int, landing_marker)), 20, (0, 255, 0), 3)
            cv2.circle(annotated_frame, tuple(map(int, landing_marker)), 4, (0, 0, 255), -1)
        landing_marker_timer -= 1
    
    # ===========================
    # 🔹 Scoreboard
    # ===========================
    cv2.rectangle(annotated_frame, (10, 10), (380, 110), (0, 0, 0), -1)
    cv2.rectangle(annotated_frame, (10, 10), (380, 110), (255, 255, 255), 2)
    
    cv2.putText(annotated_frame, f"PLAYER 1: {player1_score}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"PLAYER 2: {player2_score}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    status_color = (0, 255, 0) if rally_active else (100, 100, 100)
    cv2.circle(annotated_frame, (350, 55), 12, status_color, -1)
    
    # Scoring notification
    if scoring_event and scoring_event_timer > 0:
        text_size = cv2.getTextSize(scoring_event, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] // 2
        
        cv2.rectangle(annotated_frame, 
                     (text_x - 15, text_y - 45),
                     (text_x + text_size[0] + 15, text_y + 15), 
                     (0, 0, 0), -1)
        cv2.rectangle(annotated_frame,
                     (text_x - 15, text_y - 45),
                     (text_x + text_size[0] + 15, text_y + 15),
                     (0, 255, 0), 3)
        
        cv2.putText(annotated_frame, scoring_event, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        
        scoring_event_timer -= 1
        if scoring_event_timer == 0:
            scoring_event = None
    
    # Status display
    status_y = 140
    if rally_active:
        cv2.putText(annotated_frame, "RALLY ACTIVE", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        status_y += 25
    
    if shuttle_state['ground_frames'] > 0:
        cv2.putText(annotated_frame, f"GROUND: {shuttle_state['ground_frames']}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        status_y += 25
    
    if shuttle_state['low_position_frames'] > 0:
        cv2.putText(annotated_frame, f"LOW POS: {shuttle_state['low_position_frames']}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        status_y += 25
    
    if len(velocity_history) > 0:
        vx, vy = velocity_history[-1]
        speed = np.sqrt(vx**2 + vy**2)
        cv2.putText(annotated_frame, f"Speed: {speed:.1f}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # ===========================
    # 🔹 Write frame to output video
    # ===========================
    out.write(annotated_frame)
    
    cv2.imshow("Badminton Score Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n⚠️ Video processing interrupted by user")
        break

# ===========================
# 🔹 Cleanup
# ===========================
cap.release()
out.release()  # Release the video writer
cv2.destroyAllWindows()

print("\n" + "="*60)
print("🏸 FINAL SCORE 🏸")
print("="*60)
print(f"Player 1: {player1_score}")
print(f"Player 2: {player2_score}")
print("="*60)
print(f"✅ Video saved to: {output_path}")
print("✅ Badminton Score Detection Completed")