import os
import sys
import threading
import time

import cv2
import numpy
import numpy as np
from PyQt5 import Qt
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from MainWindow import *
import model
import tensorflow as tf
import multiprocessing


def enlarge_process(inputimage_path, scale, isusegpu, queue, ):
    if isusegpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定第一块GPU可用
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    with tf.compat.v1.Session(config=config) as session:
        session = tf.compat.v1.Session(config=config)
        sr = model.SRmodel(session, './dataset.h5', 200, 64, 2e-4, './testdataset.h5', './log', False)
        sr.load_model('./model/model_data.ckpt')
        output_image="failed"
        try:
            output_image = sr.enlarge_YCrCb(inputimage_path, scale=scale)
        finally:
            queue.put(output_image)


class signal(QObject):
    # 自定义一个信号
    my_sighal = pyqtSignal(str)

    # 定义一个发送信号的函数
    def run(self, text):
        self.my_sighal.emit(text)


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        global send
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.loadimage_button.clicked.connect(self.openimage)
        self.enlarge_button.clicked.connect(self.enlarge)
        self.inputimage_path = ""
        self.save_button.clicked.connect(self.saveImage)

        send = signal()
        send.my_sighal.connect(self.action)
        self.textBrowser.append("MFRL by xyc")

    def action(self, text):
        msg_box = QtWidgets.QMessageBox
        msg_box.information(self, 'info', text)

    def box(self, title, info):
        QMessageBox.information(self, title, info, QMessageBox.Ok)

    def proportional_zoom(self, label, image):
        if image.height() < image.width():
            width = label.width()
            height = int(image.height() * (width / image.width()))
        else:
            height = label.height()
            width = int(image.width() * (height / image.height()))
        return width, height

    def check(self, queue, ):
        global output_image
        output_image = queue.get()
        if isinstance(output_image, str):
            send.run("放大失败，请尝试禁用cuda,或者检查输入图片")
            output_image = None
            return
        send.run("图片放大成功!")
        self.textBrowser.append("<h3>图片放大成功!</h3>")
        image = QtGui.QImage(output_image, output_image.shape[1], output_image.shape[0],output_image.shape[1]*3, QtGui.QImage.Format_BGR888)
        width, height = self.proportional_zoom(self.display_show_image, image)
        image = QtGui.QPixmap(image).scaled(width, height)
        self.display_show_image.setPixmap(image)

    def saveImage(self):
        if ThreadEnlarge != None:
            if ThreadEnlarge.is_alive():
                msg_box = QtWidgets.QMessageBox
                msg_box.warning(self, '警告', '正在进行图片放大，请稍后再试')
                return
        if not isinstance(output_image, np.ndarray):
            msg_box = QtWidgets.QMessageBox
            msg_box.warning(self, '警告', '输出图片为空')
            return
        outimage_path, imgType = QFileDialog.getSaveFileName(self, "保存图片", "", "*.png;;*.jpg;;All Files(*)")
        if outimage_path=="":
            return
        cv2.imwrite(outimage_path, output_image)
        self.textBrowser.append(str.format("图片保存成功 路径：{}",outimage_path))
        self.textBrowser.append(str.format("输出图片信息：height:{} width:{}",output_image.shape[0],output_image.shape[1]))
    def openimage(self):
        if ThreadEnlarge != None:
            if ThreadEnlarge.is_alive():
                msg_box = QtWidgets.QMessageBox
                msg_box.warning(self, '警告', '正在进行图片放大，请稍后再试')
                return
        global output_image
        output_image = None
        self.old = self.inputimage_path
        self.inputimage_path, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.png;;*.jpg;;All Files(*)")
        if self.inputimage_path == "":
            self.inputimage_path = self.old
            return
        jpg = QtGui.QPixmap(self.inputimage_path)
        self.textBrowser.append(str.format("图片：{} 成功加载",self.inputimage_path))
        self.textBrowser.append(str.format("图片信息：height:{} width:{}",jpg.height(),jpg.width()))
        width, height = self.proportional_zoom(self.show_image, jpg)
        jpg = jpg.scaled(width, height)
        self.show_image.setPixmap(jpg)
        self.display_show_image.setPixmap(QtGui.QPixmap(""))

    def enlarge(self):
        if self.inputimage_path == "":
            msg_box = QtWidgets.QMessageBox
            msg_box.warning(self, '警告', '输入图片为空')
            return
        global ThreadEnlarge
        if ThreadEnlarge != None:
            if ThreadEnlarge.is_alive():
                msg_box = QtWidgets.QMessageBox
                msg_box.warning(self, '警告', '正在进行图片放大，请稍后再试')
                return
        self.display_show_image.setPixmap(QtGui.QPixmap(""))
        scale = 4
        if self.radioButton_X2.isChecked():
            scale = 2
        elif self.radioButton_X3.isChecked():
            scale = 3
        else:
            scale = 4

        isusegpu = self.cuda_check.isChecked()

        self.textBrowser.append(str.format("<h3>图片开始放大.... 倍率：{}</h3>",scale))
        n = multiprocessing.Queue()
        t3 = multiprocessing.Process(target=enlarge_process, args=(self.inputimage_path, scale, isusegpu, n,))
        t3.start()
        ThreadEnlarge = t3
        threading.Thread(target=self.check, args=(n,)).start()


if __name__ == '__main__':
    ThreadEnlarge = None
    output_image = None
    send = None
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
