import cv2
import numpy as np
import time

def calculate_optical_flow(prev_gray, next_gray, prev_pts):
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, None)

    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    return good_old, good_new

cap = cv2.VideoCapture("Sample Video.mp4")

ret, prev_frame = cap.read()
if not ret:
    print("Failed to read the vid")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)    

prev_pts = cv2.goodFeaturesToTrack(
    prev_gray,
    mask = None,
    **{
        "maxCorners": 1000,
        "qualityLevel": 0.1,
        'minDistance' : 7,
        "blockSize" : 7 
    }
)

mask = np.zeros_like(prev_frame)

while True:
    ret, next_frame = cap.read()
    if not ret:
        break

    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    good_old, good_new = calculate_optical_flow(prev_gray, next_gray, prev_pts)

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        next_frame = cv2.circle(next_frame, (int(a), int(b)), 5, (0, 0, 255), -1)

    img = cv2.add(next_frame, mask)
    cv2.imshow("Optical Flow", img)
    time.sleep(1)
    prev_gray = next_gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()