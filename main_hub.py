'''
Program Title: YOLOv5 - GKD: Real-Time Weapon Detection for Surveillance Using Enhanced YOLOv5

Programmers:    Veluz, John Ivan V.
                Flores, Julius Martin A.
                Paragoya, Vince M.
                Sembrano, Robert Vince M.

Where the Program Fits in the General System Design:
                This connects the structures of the YOLO Algorithm and acts as the main application of the whole system. The connection from back-end
                to front-end as well as the structure of the front-end is located in this program.

Date Written: November 27, 2022
Date Revised: January 30, 2023

Purpose of the Program:

The creation of this system has the primary goal of creating a real-time weapon detection system that is able to
detect Filipino made and improvised weapon using the latest algorithm along with some modifications to the structure
to make the detection faster and more accurate even when the object to be detected is small and even when the lighting
condition is different.

The system used the following Data Structures, Algorithms, and Control:

Data Structures, Algorithms, and Control:
    Algorithms: YOLO Algorithm
    Control: If Else Statement, Try Catch Statement, Iteration
    Data Structures: Lists

'''

from PyQt5 import QtCore, QtGui, QtWidgets

import argparse
import os
import sys
from pathlib import Path
import cv2

import torch

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import os
import shutil

import time
import glob

#Initialization of the input and output panels.
#Directly connected to the YOLOv5
class ScaledPixmapLabel(QtWidgets.QLabel):
    def paintEvent(self, event):
        if self.pixmap():
            pm = self.pixmap()
            originalRatio = pm.width() / pm.height()
            currentRatio = self.width() / self.height()
            if originalRatio != currentRatio:
                qp = QtGui.QPainter(self)
                pm = self.pixmap().scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                rect = QtCore.QRect(0, 0, pm.width(), pm.height())
                rect.moveCenter(self.rect().center())
                qp.drawPixmap(rect, pm)
                return
        super().paintEvent(event)

class Ui_MainWindow(object):
    #Initialization of local variables
    loaded_model = None #model status
    cam_running = None  #camera status
    vid_running = None  #video status
    opt = None          #end of detection
    
    num_stop = 1 
    output_folder = 'output/'
    vid_writer = None
    
    openfile_name_model = None
    vid_name = None
    img_name = None
    cap_statue = None
    save_dir = None
    img_over = None

    timer = QtCore.QTimer()
    
    cap_video = 0
    flag = 0
    img = []
    video_stream = None
    
    alreadystarted_cam = None
    alreadystarted_vid = None
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)
    
    curr_framecount = 0
    curr_fps = 30
    curr_maxframe = 0
    
    recording = None
    saving = None
    
    thistime = None
    selected = None
    
    def init_slots(self):
        #Button functions
        self.button_load_model.clicked.connect(self.load_model)
        self.button_detect.clicked.connect(self.open_detection)
        self.button_load_image.clicked.connect(self.open_img)
        self.button_load_video.clicked.connect(self.open_vid)
        self.button_load_camera.clicked.connect(self.open_cam)
        self.button_record.clicked.connect(self.record)
        self.button_save.clicked.connect(self.save)

        self.button_load_image.setDisabled(True)
        self.button_load_video.setDisabled(True)
        self.button_load_camera.setDisabled(True)
        
        pass

    def open_img(self):
        #START DETECTION FOR IMAGES
        self.selected = "img"
        self.button_record.setDisabled(True)
        self.button_save.setDisabled(True)
        # try except
        self.cam_running = False
        self.vid_running = False
        dir = 'detections/images'

        #Create a folder in the file path if folder does not exist - selects specific file format for images
        if not os.path.exists(dir):
            
            os.makedirs(dir)
        try:
            #Selection of file to detect
            self.img_name, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "sample inputs", "*.jpg ; *.png ; All Files(*)")
        except OSError as reason:
            print(str(reason))
        else:
            #Process the detection
            im0 = cv2.imread(self.img_name)
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            FlippedImage = im0
            image = QtGui.QImage(FlippedImage, FlippedImage.shape[1],FlippedImage.shape[0], FlippedImage.shape[1] * 3, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap(image)
            

            self.image_box_1.setPixmap(QtGui.QPixmap(pix))
            
            results = self.loaded_model(self.img_name)
            results.ims
            results.render()
            for im in results.ims:

                FlippedImage = im
                image = QtGui.QImage(FlippedImage, FlippedImage.shape[1],FlippedImage.shape[0], FlippedImage.shape[1] * 3, QtGui.QImage.Format_RGB888)
                pix = QtGui.QPixmap(image)

                FlippedImage = cv2.cvtColor(FlippedImage, cv2.COLOR_BGR2RGB)
                cv2.imwrite("detections/images/" + str(time.time()) + ".jpg", FlippedImage)
                self.image_box_2.setPixmap(QtGui.QPixmap(pix))
        pass

    def open_vid(self):
        #START DETECTION FOR IMAGES
        self.selected = "video"
        self.cam_running = False
        self.button_record.setDisabled(False)
        self.button_save.setDisabled(False)
        try:
            #Selection of file to detect
            self.vid_name, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Video", "sample inputs", "*.mp4;*.mkv;All Files(*)")
        except OSError as reason:
            print(str(reason))
        else:
            #Error catcher if file does not exist
            if not self.vid_name:
                pass
            else:
                # Process the detection
                # Files: video, frames per video
                cap = cv2.VideoCapture(self.vid_name)

                if not cap.isOpened(): 
                    print("could not open :",self.vid_name)
                    return
                    
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps    = cap.get(cv2.CAP_PROP_FPS)
                
                print(length, width, height, fps)
                
                self.curr_fps = fps
                self.curr_maxframe = length
                self.curr_framecount = 0
                
                dir = 'detections/cache'
                if os.path.exists(dir):
                    shutil.rmtree(dir)
                    os.makedirs(dir)
                else:
                    os.makedirs(dir)
                
                self.thistime = str(time.time())
                dir = 'detections/video_cache/' + self.thistime
                if not os.path.exists(dir):
                    #shutil.rmtree(dir)
                    os.makedirs(dir)
                
                self.vid_running = True
                self.video_stream = cv2.VideoCapture(self.vid_name)
                self.timer.start(20)
                
    def open_cam(self):
        #REAL-TIME DETECTION
        #accepts values 0 to n for connected device. 0 for default = webcam.
        self.selected = "cam"
        self.cam_running = True
        self.vid_running = False
        self.button_record.setDisabled(False)
        self.button_save.setDisabled(False)
        dir = 'detections/cache'
        if os.path.exists(dir):
            shutil.rmtree(dir)
            os.makedirs(dir)
        else:
            os.makedirs(dir)
        
        self.thistime = str(time.time())
        dir = 'detections/cam_cache/' + self.thistime #file path for frame extraction
        if not os.path.exists(dir):
            os.makedirs(dir)
                    
        if not self.alreadystarted_cam: #camera status
            self.alreadystarted_cam = True
            self.cap_video = cv2.VideoCapture(0)
            self.timer.start(20)
            
    def show_video(self):

        #OUTPUT PANEL TO SHOW DETECTION INSTANTLY
        result, image = None, None
        if self.vid_running: #save each frame
            result, image = self.video_stream.read()
        if result:
            im0 = image
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            FlippedImage = im0
            image = QtGui.QImage(FlippedImage, FlippedImage.shape[1],FlippedImage.shape[0], FlippedImage.shape[1] * 3, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap(image)

            cv2.imwrite("data/images/temp_vid.jpg", FlippedImage)
            self.image_box_1.setPixmap(QtGui.QPixmap(pix))
            #process output in the output panel
            results = self.loaded_model('data/images/temp_vid.jpg')#.save()
            results.ims
            results.render()
            for im in results.ims:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                
                FlippedImage = im
                image = QtGui.QImage(FlippedImage, FlippedImage.shape[1],FlippedImage.shape[0], FlippedImage.shape[1] * 3, QtGui.QImage.Format_RGB888)
                pix = QtGui.QPixmap(image)
            
                if self.recording:
                    FlippedImage = cv2.cvtColor(FlippedImage, cv2.COLOR_BGR2RGB)
                    
                    cv2.imwrite("detections/cache/" + str(time.time()) + ".jpg", FlippedImage)
                    cv2.imwrite('detections/video_cache/' + self.thistime + "/" + str(time.time()) + ".jpg", FlippedImage)
                self.image_box_2.setPixmap(QtGui.QPixmap(pix))
            self.curr_framecount += 1
            if self.curr_framecount == self.curr_maxframe:
                self.vid_running = False
                #Save Video
    def show_video_cam(self):
        # reading the input using the camera
        result, image = None, None
        if self.cam_running:
            result, image = self.cap_video.read()

        # If image will detected without any error, 
        # show result
        if result:
            im0 = image
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            FlippedImage = cv2.flip(im0, 1)
            image = QtGui.QImage(FlippedImage, FlippedImage.shape[1],FlippedImage.shape[0], FlippedImage.shape[1] * 3, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap(image)
 
            cv2.imwrite("data/images/temp_cam.jpg", FlippedImage)
            self.image_box_1.setPixmap(QtGui.QPixmap(pix))
            
            results = self.loaded_model('data/images/temp_cam.jpg')#.save()
            results.ims
            results.render()
            for im in results.ims:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
               
                FlippedImage = im
                image = QtGui.QImage(FlippedImage, FlippedImage.shape[1],FlippedImage.shape[0], FlippedImage.shape[1] * 3, QtGui.QImage.Format_RGB888)
                pix = QtGui.QPixmap(image)
                
                if self.recording:
                    FlippedImage = cv2.cvtColor(FlippedImage, cv2.COLOR_BGR2RGB)
                    
                    cv2.imwrite("detections/cache/" + str(time.time()) + ".jpg", FlippedImage)
                    cv2.imwrite('detections/cam_cache/' + self.thistime + "/" + str(time.time()) + ".jpg", FlippedImage)
                self.image_box_2.setPixmap(QtGui.QPixmap(pix))
                
    def record(self):
        # Buttons for recording - changing status
        _translate = QtCore.QCoreApplication.translate
        self.recording = not self.recording
        if self.recording:
            self.button_record.setText(_translate("MainWindow", "RECORDING"))
        else:
            self.button_record.setText(_translate("MainWindow", "  RECORD"))
        
    def save(self):
        # Function to save the output
        # can save each frame
        if self.selected == 'video':
            dir = 'detections/videos'
            if not os.path.exists(dir):
                
                os.makedirs(dir)
            
            try:
                dimensions = None
                for filename in glob.glob('detections/cache/*.jpg'):
                    img = cv2.imread(filename)
                    dimensions = img.shape
                    break
                print("Saving...")
                frameSize = (dimensions[1], dimensions[0])
                
                out = cv2.VideoWriter('detections/videos/' + str(time.time()) + ".mp4",cv2.VideoWriter_fourcc(*'mpv4'), int(self.curr_fps), frameSize)
                # selects file path and created folder if it doesn't exists - saves in mp4 and jpg
                for filename in glob.glob('detections/cache/*.jpg'):
                    print(filename)
                    
                    img = cv2.imread(filename)
                    
                    out.write(img)
                print("Saved!")
                out.release()
            except Exception as e:
                print(e)
        if self.selected == 'cam':
            dir = 'detections/camera'
            if not os.path.exists(dir):
                
                os.makedirs(dir)
            
            try:
                dimensions = None
                for filename in glob.glob('detections/cache/*.jpg'):
                    img = cv2.imread(filename)
                    dimensions = img.shape
                    break
                print("Saving...")
                frameSize = (dimensions[1], dimensions[0])
                # saves the video recording from output
                out = cv2.VideoWriter('detections/camera/' + str(time.time()) + ".mp4",cv2.VideoWriter_fourcc(*'mpv4'), int(self.curr_fps), frameSize)
                # selects file path and created folder if it doesn't exists
                for filename in glob.glob('detections/cache/*.jpg'):
                    print(filename)
                    
                    img = cv2.imread(filename)

                    out.write(img)
                print("Saved!")
                out.release()
            except Exception as e:
                print(e)
        
    def load_model(self):
        # Opening any YOLO model - only accepts .pt files otherwise error will appear
        try:
            self.openfile_name_model, some = QtWidgets.QFileDialog.getOpenFileName(None, 'yolov5.pt', 'weights', "*.pt")
        except OSError as reason:
            print(str(reason))
        else:
            if self.openfile_name_model:
                print(self.openfile_name_model)
                
                try:
                    self.loaded_model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.openfile_name_model, force_reload=True)
                except:
                    self.loaded_model = torch.hub.load(r'D:\yolo\yolov5-master\weights', 'custom', path=self.openfile_name_model, force_reload=True)
               
                self.output_box.append("Model loading complete!")
                self.button_load_image.setDisabled(False)
                self.button_load_video.setDisabled(False)
                self.button_load_camera.setDisabled(False)
            else:
                
                self.output_box.append("No files selected")
        

    def open_detection(self): 
        #file path for opening the recorded outputs     
        os.startfile(r'C:\Users\Ivan\Documents\4th Year\yolo-main\detections')

    def initialize(self):
        self.init_slots()
        self.timer.timeout.connect(self.show_video_cam)
        self.timer.timeout.connect(self.show_video)

    def setupUi(self, MainWindow):
        #GUI COMPONENTS
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1100, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
       
        MainWindow.setStyleSheet("background-color: #6F90AF;")
        self.centralwidget.setObjectName("centralwidget")

        font = QtGui.QFont() # font design for the title
        font.setFamily("Poppins")
        font.setPointSize(16)
        font.setBold(True)
        font2 = QtGui.QFont() # font design for the 'Input' and 'Output'
        font2.setFamily("Poppins")
        font2.setPointSize(18)
        font2.setBold(True)
        font3 = QtGui.QFont() # font design for the buttons
        font3.setPointSize(10)
        font3.setFamily("Poppins")
        font3.setBold(True)
        font4 = QtGui.QFont() # font design for the status
        font4.setPointSize(10)
        font4.setFamily("Poppins")
        font4.setBold(True)
        font5 = QtGui.QFont() # font design for the status
        font5.setPointSize(12)
        font5.setFamily("Poppins")
        font5.setBold(True)

        self.button_load_model = QtWidgets.QPushButton(self.centralwidget)
        self.button_load_model.setGeometry(QtCore.QRect(15, 120, 144, 31))
        self.button_load_model.setFont(font3)
        self.button_load_model.setStyleSheet('QPushButton {background-color: WHITE;}')
        self.button_load_model.setObjectName("button_load_model")
        self.button_load_image = QtWidgets.QPushButton(self.centralwidget)
        self.button_load_image.setGeometry(QtCore.QRect(15, 160, 144, 31))
        self.button_load_image.setFont(font3)
        self.button_load_image.setStyleSheet('QPushButton {background-color: WHITE;}')
        self.button_load_image.setObjectName("button_load_image")
        self.button_load_video = QtWidgets.QPushButton(self.centralwidget)
        self.button_load_video.setGeometry(QtCore.QRect(15, 200, 144, 31))
        self.button_load_video.setFont(font3)
        self.button_load_video.setStyleSheet('QPushButton {background-color: WHITE;}')
        self.button_load_video.setObjectName("button_load_video")
        self.button_load_camera = QtWidgets.QPushButton(self.centralwidget)
        self.button_load_camera.setGeometry(QtCore.QRect(15, 240, 144, 31))
        self.button_load_camera.setFont(font3)
        self.button_load_camera.setStyleSheet('QPushButton {background-color: WHITE;}')
        self.button_load_camera.setObjectName("button_load_camera")
        self.button_record = QtWidgets.QPushButton(self.centralwidget)
        self.button_record.setGeometry(QtCore.QRect(15, 280, 144, 31))
        self.button_record.setFont(font3)
        self.button_record.setIcon(QIcon('UI/recording.png'))
        self.button_record.setStyleSheet('QPushButton {background-color: WHITE;}')
        self.button_record.setObjectName("button_record")
        self.output_box = QtWidgets.QTextEdit(self.centralwidget)
        self.output_box.setGeometry(QtCore.QRect(15, 450, 144, 121))
        self.output_box.setFont(font4)
        self.output_box.setObjectName("output_box")
        self.output_box.setReadOnly(True)
        self.output_box.setStyleSheet("background-color: white;")
        self.image_box_1 = ScaledPixmapLabel(MainWindow)
        self.image_box_1.setGeometry(QtCore.QRect(170, 120, 450, 450))
        self.image_box_1.setFrameShape(QtWidgets.QFrame.Box)
        self.image_box_1.setText("")
        self.image_box_1.setObjectName("image_box_1")
        self.image_box_1.setScaledContents(True)
        self.image_box_1.setStyleSheet("background-color: white;")
        self.image_box_2 = ScaledPixmapLabel(MainWindow)
        self.image_box_2.setGeometry(QtCore.QRect(630, 120, 450, 450))
        self.image_box_2.setFrameShape(QtWidgets.QFrame.Box)
        self.image_box_2.setText("")
        self.image_box_2.setObjectName("image_box_2")
        self.image_box_2.setScaledContents(True)
        self.image_box_2.setStyleSheet("background-color: white;")
        self.image_label_1 = QtWidgets.QLabel(self.centralwidget)
        self.image_label_1.setGeometry(QtCore.QRect(330, 80, 144, 31))
        self.image_label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label_1.setFont(font2)
        self.image_label_1.setObjectName("image_label_1")
        self.image_label_1.setStyleSheet("color: white;")
        self.image_label_2 = QtWidgets.QLabel(self.centralwidget)
        self.image_label_2.setGeometry(QtCore.QRect(780, 80, 144, 31))
        self.image_label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label_2.setFont(font2)
        self.image_label_2.setObjectName("image_label_2")
        self.image_label_2.setStyleSheet("color: white;")
        self.image_label_3 = QtWidgets.QLabel(self.centralwidget)
        self.image_label_3.setGeometry(QtCore.QRect(70, 20, 950, 41))
        self.image_label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label_3.setFont(font)
        self.image_label_3.setObjectName("image_label_3")
        self.image_label_3.setStyleSheet("color: white;")
        self.image_label_4 = QtWidgets.QLabel(self.centralwidget)
        self.image_label_4.setGeometry(QtCore.QRect(15, 420, 60, 25))
        self.image_label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label_4.setFont(font5)
        self.image_label_4.setObjectName("image_label_3")
        self.image_label_4.setStyleSheet("color: white;")
        self.button_save = QtWidgets.QPushButton(self.centralwidget)
        self.button_save.setGeometry(QtCore.QRect(15, 320, 144, 31))
        self.button_save.setFont(font3)
        self.button_save.setIcon(QIcon('UI/save.png'))
        self.button_save.setStyleSheet('QPushButton {background-color: WHITE;}')
        self.button_save.setObjectName("button_save")
        self.button_detect = QtWidgets.QPushButton(self.centralwidget)
        self.button_detect.setGeometry(QtCore.QRect(15, 360, 144, 31))
        self.button_detect.setFont(font3)
        self.button_detect.setIcon(QIcon('UI/search.png'))
        self.button_detect.setStyleSheet('QPushButton {background-color: WHITE;}')
        self.button_detect.setObjectName("button_detect")
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.initialize()
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        #Initialization for GUI Components
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Weapon Detection System"))
        self.button_load_model.setText(_translate("MainWindow", "LOAD MODEL"))
        self.button_load_image.setText(_translate("MainWindow", "LOAD IMAGE"))
        self.button_load_video.setText(_translate("MainWindow", "LOAD VIDEO"))
        self.button_load_camera.setText(_translate("MainWindow", "LOAD CAMERA"))
        self.button_record.setText(_translate("MainWindow", "  RECORD"))
        self.image_label_1.setText(_translate("MainWindow", "INPUT"))
        self.image_label_2.setText(_translate("MainWindow", "OUTPUT"))
        self.image_label_3.setText(_translate("MainWindow", "YOLOv5 GKD: Real-Time Weapon Detection for Surveillance using Enhanced YOLOv5"))
        self.image_label_4.setText(_translate("MainWindow", "Status:"))
        self.button_save.setText(_translate("MainWindow", "  SAVE"))
        self.button_detect.setText(_translate("MainWindow", " DETECTIONS"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
