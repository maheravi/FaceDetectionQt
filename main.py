import sys
from PySide6.QtWidgets import *
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import *
import cv2

from PySide6.QtGui import QPixmap, QImage


def convertCvImage2QtImage(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    height, width, channel = cv_img.shape
    bytesPerLine = 3 * width
    qImg = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)


class FaceDetector(QThread):
    signal_success_process = Signal(object)

    def __init__(self, image_path):
        super(FaceDetector, self).__init__()
        self.image_path = image_path

    def run(self):
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        image = cv2.imread(self.image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(image_gray, 1.3)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 8)

        # cv2.imwrite('output.png', image)
        self.signal_success_process.emit(image)


class Main(QWidget):
    def __init__(self):
        super(Main, self).__init__()

        loader = QUiLoader()
        self.ui = loader.load("form.ui")
        self.ui.browse.clicked.connect(self.openImage)
        self.ui.start_face.clicked.connect(self.startFaceDetection)
        self.ui.webcam.clicked.connect(self.webcam)

        self.ui.show()

    def openImage(self):
        image_path = QFileDialog.getOpenFileName(self, 'Open Your Picture')
        self.image_path = image_path[0]
        self.ui.destination.setText(self.image_path)
        my_pixmap = QPixmap(self.image_path)
        self.ui.label.setPixmap(my_pixmap)

    def startFaceDetection(self):
        self.face_detector = FaceDetector(self.image_path)
        self.face_detector.start()
        self.face_detector.signal_success_process.connect(self.showOutput)

    # slot
    def showOutput(self, image):

        my_pixmap = convertCvImage2QtImage(image)
        # my_pixmap = QPixmap('output.png')
        self.ui.label.setPixmap(my_pixmap)

    def webcam(self):
        my_video = cv2.VideoCapture(0)
        while True:

            validation, frame = my_video.read()

            if validation is not True:
                break
            face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(frame_gray, 1.3)

            for i, face in enumerate(faces):
                x, y, w, h = face

                frame_face = frame[y:y + h, x:x + w]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 8)

            cv2.waitKey(10)
            self.showOutput(frame)


if __name__ == "__main__":
    app = QApplication([])
    window = Main()
    sys.exit(app.exec())
