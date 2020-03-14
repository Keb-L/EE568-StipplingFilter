# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import lloyd_relaxation

class Ui_MainWindow(object):
    L = lloyd_relaxation([])

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 502)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 10, 341, 321))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label.setFont(font)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(420, 10, 341, 321))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_2.setFont(font)
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(30, 350, 191, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.chkGradient = QtWidgets.QCheckBox(self.centralwidget)
        self.chkGradient.setGeometry(QtCore.QRect(670, 380, 70, 17))
        self.chkGradient.setObjectName("chkGradient")
        self.chkDots = QtWidgets.QCheckBox(self.centralwidget)
        self.chkDots.setGeometry(QtCore.QRect(670, 400, 70, 17))
        self.chkDots.setObjectName("chkDots")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(260, 350, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.sldStipples = QtWidgets.QSlider(self.centralwidget)
        self.sldStipples.setGeometry(QtCore.QRect(340, 380, 261, 22))
        self.sldStipples.setMinimum(100)
        self.sldStipples.setMaximum(100000)
        self.sldStipples.setSingleStep(1000)
        self.sldStipples.setPageStep(5000)
        self.sldStipples.setProperty("value", 1000)
        self.sldStipples.setOrientation(QtCore.Qt.Horizontal)
        self.sldStipples.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.sldStipples.setObjectName("sldStipples")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(270, 390, 71, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(650, 360, 47, 13))
        self.label_5.setObjectName("label_5")
        self.spnIter = QtWidgets.QSpinBox(self.centralwidget)
        self.spnIter.setGeometry(QtCore.QRect(350, 420, 42, 22))
        self.spnIter.setMinimum(1)
        self.spnIter.setMaximum(25)
        self.spnIter.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.spnIter.setObjectName("spnIter")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(270, 420, 71, 16))
        self.label_6.setObjectName("label_6")
        self.btnProcess = QtWidgets.QPushButton(self.centralwidget)
        self.btnProcess.setGeometry(QtCore.QRect(30, 400, 191, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.btnProcess.setFont(font)
        self.btnProcess.setObjectName("btnProcess")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(610, 380, 41, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setFrameShape(QtWidgets.QFrame.Box)
        self.label_7.setObjectName("label_7")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # ---------------------------------------------------
        # when user click the button, we run self.myloadingimage
        self.pushButton.clicked.connect(self.myloadingimage)
        self.btnProcess.clicked.connect(self.processimage)

        # ---------------------------------------------------
        # when user change the value using the scroller bar, we run self.myimageprocess
        self.sldStipples.valueChanged.connect(self.updateState)
        self.chkGradient.stateChanged.connect(self.updateState)
        self.chkDots.stateChanged.connect(self.updateState)
        self.spnIter.valueChanged.connect(self.updateState)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "LoadedImage"))
        self.label_2.setText(_translate("MainWindow", "ProcessedImage"))
        self.pushButton.setText(_translate("MainWindow", "Load an Image"))
        self.chkGradient.setText(_translate("MainWindow", "Gradient"))
        self.chkDots.setText(_translate("MainWindow", "Dots"))
        self.label_3.setText(_translate("MainWindow", "Parameters"))
        self.label_4.setText(_translate("MainWindow", "Stipple Count"))
        self.label_5.setText(_translate("MainWindow", "Style"))
        self.label_6.setText(_translate("MainWindow", "Max Iterations"))
        self.btnProcess.setText(_translate("MainWindow", "Process Image"))
        self.label_7.setText(_translate("MainWindow", str(self.sldStipples.value())))

    def updateState(self):
        self.L.N = self.sldStipples.value()
        self.L.grad = self.chkGradient.isChecked()
        self.L.dots = self.chkDots.isChecked()
        self.L.maxiter = self.spnIter.value()

  # ---------------------------------------------------
    # implmentation for converting image representation to Qt
    def myCV2Qt(self, img):
        height, width, channel = img.shape
        BytePerLine = width*channel
        qimg = QtGui.QImage(img.data,width,height,BytePerLine,QtGui.QImage.Format_RGB888)
        qimg = QtGui.QPixmap(qimg)
        return qimg

    # ---------------------------------------------------
    # implmentation for loading and displaying the image.
    def myloadingimage(self):
        fullpath, file_ext = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, 'Caption: Select an Image', '/home/kevin', '*.bmp *.jpg *.png')
        self.img = cv2.imread(fullpath)
        qt_img =  self.myCV2Qt(self.img)
        qt_img = qt_img.scaled(self.label.width(),self.label.height())
        self.label.setPixmap(qt_img)

    # ---------------------------------------------------
    # implmentation for your image processing
    def myimageprocess(self):
        value = self.horizontalScrollBar.value()
        # [0,100]
        value = float(value)/100
        self.img = self.img.astype('double')
        result = self.img*value
        result = np.clip(result,0,255)
        result = result.astype('uint8')
        qt_img =  self.myCV2Qt(result)
        qt_img = qt_img.scaled(self.label_2.width(),self.label_2.height())
        self.label_2.setPixmap(qt_img) 

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
