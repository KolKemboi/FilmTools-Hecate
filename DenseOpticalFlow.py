import numpy as np
import cv2
import time

def draw_flow(rgb_img ,img, flow, steps = 16):
    h, w = img.shape[:2]
    y, x = np.mgrid[steps//2:h:steps, steps//2:w:steps].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
   ### flat = np.ones((img.shape[0], img.shape[1])) * 255

    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))
    

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(rgb_img, (x1, y1), 1, (0, 255, 0), -1)
        ##cv2.circle(flat, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr




def main(filename: str):
    cap = cv2.VideoCapture(filename)

    ret, prev_frame = cap.read()

    if not ret:
        print("ERROR::FILE READING")
        cap.release()
        exit()

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not ret:
            break

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        prev_gray = gray

        flow = draw_flow(frame, gray, flow)

        cv2.imshow("Optical Flow Vectors", flow)
        ##cv2.imshow("Optical Flow Vectors", flat)
        ##time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("Sample Video.mp4")

    