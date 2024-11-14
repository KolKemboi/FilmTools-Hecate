import numpy as np
import cv2
import time

def draw_flow(img, flow, steps = 16):
    pass


def draw_hsv(flow):
    pass


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

        flow = draw_flow(gray, flow)
        hsv = draw_hsv(flow)

        cv2.imshow("Optical Flow Vectors", flow)
        cv2.imshow("Optical Flow HSV", hsv)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    