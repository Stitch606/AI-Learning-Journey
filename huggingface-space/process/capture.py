import cv2


class capture:
    def __init__(self, video):
        self.video = video

        self.cap = cv2.VideoCapture(self.video)
        if not self.cap.isOpened():
            print("the video is broken")

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()
