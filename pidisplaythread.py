# Thread dédié a l'affichage d'un flux video

from threading import Thread
from threading import Lock
import cv2

class PiDisplayThread :
    def __init__(self) :
        self.frame = None
        self.isRunning = False
        self.window_name = None
        self.lock = Lock()

    def run(self) :
        while self.isRunning :
            if self.frame is not None :
                cv2.imshow(self.window_name, self.frame)
                cv2.waitKey(1)

    def start(self, window_name) :
        self.window_name = window_name
        self.isRunning = True
        t = Thread(target=self.run, args=())
        t.start()
        return self

    def updateFrame(self, frame) :
        self.lock.acquire()
        self.frame = frame
        self.lock.release()

    def stop(self) :
        self.isRunning = False