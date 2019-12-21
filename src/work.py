import cv2
from src.fingerDetect import Detector
from src.util import *


def main():

    is_hand_hist_created = False
    capture = cv2.VideoCapture(0)
    hand_hist = None
    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        detect_hand = Detector(frame)

        if pressed_key & 0xFF == ord('z'):
            is_hand_hist_created = True
            hand_hist = detect_hand.hand_histogram()

        if is_hand_hist_created:
            detect_hand.manage_image_opr(hand_hist)
        else:
            frame = detect_hand.draw_rect()

        cv2.imshow("Live Feed",rescale_frame(cv2.flip(frame,1)))

        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


def mainColor():
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        detect_hand = Detector(frame)

        handPart=detect_hand.detectByColor()
        handPart=detect_hand.segmentByColor(handPart)
        detect_hand.manage_image_opr_color(handPart)
        cv2.imshow("001", frame)

        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
    # mainColor()
