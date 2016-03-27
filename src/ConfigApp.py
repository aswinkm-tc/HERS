__author__ = 'luz1f3r'
#!usr/bin/python
#--------------HERS(Human Emotion Recognition System------------------
import sys
import cv2
import pyaudio
import threading
import atexit
import numpy as np
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PyQt4 import QtGui,QtCore
import detectfacial

#Global Auth Variable
Trigger=0
emotion=[0,0,0,0,0]
#Auth Function
class Login(QtGui.QDialog):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        name=QtGui.QLabel('User Name:',self)
        name.sizeHint()
        name.move(0,0)
        self.textName = QtGui.QLineEdit(self)
        passwd=QtGui.QLabel('Password:',self)
        self.textPass = QtGui.QLineEdit(self)
        self.textPass.setEchoMode(QtGui.QLineEdit.Password)
        buttonLogin = QtGui.QPushButton('Login', self)
        buttonLogin.clicked.connect(self.handleLogin)

        layout = QtGui.QGridLayout(self)
        layout.addWidget(name,0,0)
        layout.addWidget(self.textName,0,1,1,2)
        layout.addWidget(passwd,1,0)
        layout.addWidget(self.textPass,1,1,1,2)
        layout.addWidget(buttonLogin,2,1)

    def handleLogin(self):
        if (self.textName.text() == 'foo' and
            self.textPass.text() == 'bar'):
            global Trigger
            self.accept()
            if Trigger==1:
                self.newWindow=imageWindow()
                self.newWindow.setGeometry(0,0,940,600)
                self.newWindow.show()
            elif Trigger==2:
                self.newWindow=audioWindow()
                self.newWindow.setGeometry(0,0,940,600)
                self.newWindow.show()
        else:
            QtGui.QMessageBox.warning(self, 'Error', 'Incorrect login credentials')

#Image Configuration Utility Class
class imageWindow(QtGui.QWidget):

    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.imgWindow()

    #Methods
    def imgWindow(self):
        #Buttons
        openImg=QtGui.QPushButton('Open Image',self)
        openImg.move(0,100)
        openImg.resize(120,30)
        openImg.setStatusTip('Open an Image from Computer')
        openImg.clicked.connect(self.imageOpen)

        captureImg=QtGui.QPushButton('Capture Image',self)
        captureImg.move(0,140)
        captureImg.resize(120,30)
        captureImg.setStatusTip('Capture an Image from Camera')
        captureImg.clicked.connect(self.imgCapture)

        processImg=QtGui.QPushButton('Process Image',self)
        processImg.move(0,180)
        processImg.resize(120,30)
        processImg.setStatusTip('Process the current Image')
        processImg.clicked.connect(self.dummy1)

        make=QtGui.QPushButton('Make sample',self)
        make.move(0,220)
        make.resize(120,30)
        make.setStatusTip('Sample the Image and Show result')
        make.clicked.connect(self.quit)

        finish=QtGui.QPushButton('Finish',self)
        finish.move(0,260)
        finish.resize(120,30)
        finish.setStatusTip('Write changes to Disk')
        finish.clicked.connect(self.quit)

        #Emotion_Selectors
        select_neutral=QtGui.QRadioButton('Neutral',self)
        select_neutral.move(150,100)
        select_neutral.toggled.connect(self.neutral)
        select_happy=QtGui.QRadioButton('Happy',self)
        select_happy.move(150,120)
        select_happy.toggled.connect(self.happy)
        select_sad=QtGui.QRadioButton('Sad',self)
        select_sad.move(150,140)
        select_sad.toggled.connect(self.sad)
        select_surprise=QtGui.QRadioButton('Surprised',self)
        select_surprise.move(150,160)
        select_surprise.toggled.connect(self.surprise)
        select_anger=QtGui.QRadioButton('Angry',self)
        select_anger.move(150,180)
        select_anger.toggled.connect(self.angry)
        select_fear=QtGui.QRadioButton('Scared',self)
        select_fear.move(150,200)
        select_fear.toggled.connect(self.fear)

        #progress Bar
        progress=QtGui.QProgressBar(self)
        progress.setGeometry(0,550,940,25)

        #displayImage
        self.imageLabel=QtGui.QLabel(self)
        self.imageLabel.setGeometry(300,10,640,480)
        self.imageLabel.setStyleSheet('border: 1px solid black')

    def make_sample(self):
        file_name=""
        cv2.imwrite()

    def happy(self):
        self.name='data/'+'happy/'

    def neutral(self):
        self.name='data/'+'neutral/'

    def sad(self):
        self.name='data/'+'sad/'

    def angry(self):
        self.name='data/'+'anger/'

    def surprise(self):
        self.name='data/'+'surprise/'

    def fear(self):
        self.name='data/'+'fear/'

    #Open Image From Computer
    def imageOpen(self):
        filename=['']
        for name in QtGui.QFileDialog.getOpenFileNames(self,'Open Image'):
            name=str(name)
            print name
            filename.append(name)
        img=cv2.imread(name)
        img=cv2.cvtColor(img,cv2.cv.CV_BGR2RGB)
        img=QtGui.QImage(img, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        self.imgPix=QtGui.QPixmap.fromImage(img)
        self.imageLabel.setPixmap(self.imgPix)

    #Capture Image Using Camera
    def imgCapture(self):
        self.cam=cv2.VideoCapture(0)
        ret,img=self.cam.read()
        img=cv2.cvtColor(img,cv2.cv.CV_BGR2RGB)
        del(self.cam)
        img=QtGui.QImage(img, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        self.imgPix=QtGui.QPixmap.fromImage(img)
        self.imageLabel.setPixmap(self.imgPix)

    def quit(self):
        self.close()

    #dummy modules
    def dummy1(self):
        print "blah"

    def dummy2(self):
        print "blah"


#Audio Configuartion Utility Class
class audioWindow(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.audWindow()

    def audWindow(self):
        #Buttons
        openAudio=QtGui.QPushButton('Open Audio',self)
        openAudio.move(0,100)
        openAudio.resize(120,30)
        openAudio.setStatusTip('Open an Audio from Computer')
        openAudio.clicked.connect(self.audioOpen)

        captureAudio=QtGui.QPushButton('Capture Audio',self)
        captureAudio.move(0,140)
        captureAudio.resize(120,30)
        captureAudio.setStatusTip('Capture an Audio from Microphone')
        captureAudio.clicked.connect(self.dummy2)

        processAudio=QtGui.QPushButton('Process Audio',self)
        processAudio.move(0,180)
        processAudio.resize(120,30)
        processAudio.setStatusTip('Process the current Audio')
        processAudio.clicked.connect(self.dummy1)

        make=QtGui.QPushButton('Make sample',self)
        make.move(0,220)
        make.resize(120,30)
        make.setStatusTip('Sample the Audio and Show result')
        make.clicked.connect(self.dummy2)

        finish=QtGui.QPushButton('Finish',self)
        finish.move(0,260)
        finish.resize(120,30)
        finish.setStatusTip('Write changes to Disk')
        finish.clicked.connect(self.quit)

        #displayFeatures
        energy_title=QtGui.QLabel('Energy',self)
        energy_title.setGeometry(110,320,100,20)
        energy_progress=QtGui.QProgressBar(self)
        energy_progress.setGeometry(20,340,240,20)
        pitch_title=QtGui.QLabel('Pitch',self)
        pitch_title.setGeometry(110,360,100,20)
        pitch_progress=QtGui.QProgressBar(self)
        pitch_progress.setGeometry(20,380,240,20)
        mfcc_title=QtGui.QLabel('MFCC',self)
        mfcc_title.setGeometry(110,400,100,20)
        mfcc_progress=QtGui.QProgressBar(self)
        mfcc_progress.setGeometry(20,420,240,20)

        #Emotion_Selectors
        select_neutral=QtGui.QRadioButton('Neutral',self)
        select_neutral.move(150,100)
        select_happy=QtGui.QRadioButton('Happy',self)
        select_happy.move(150,120)

        #select_happy.toggled.connect(self.happy)
        select_sad=QtGui.QRadioButton('Sad',self)
        select_sad.move(150,140)
        select_surprise=QtGui.QRadioButton('Surprised',self)
        select_surprise.move(150,160)
        select_anger=QtGui.QRadioButton('Angry',self)
        select_anger.move(150,180)
        select_fear=QtGui.QRadioButton('Scared',self)
        select_fear.move(150,200)

        #displayImage
        self.imageLabel=QtGui.QLabel(self)
        self.imageLabel.setGeometry(300,10,640,480)
        self.imageLabel.setStyleSheet('border: 1px solid black')

        #progress Bar
        progress=QtGui.QProgressBar(self)
        progress.setGeometry(0,550,940,25)

    def audioOpen(self):
         name=QtGui.QFileDialog.getOpenFileName(self,'Open Image')
         name=str(name)

    def quit(self):
        self.close()

     #dummy modules
    def dummy1(self):
        print "blah"

    def dummy2(self):
        print "blah"

#Audio Recording and processing
class MicrophoneRecorder(object):
    def __init__(self, rate=4000, chunksize=1024):
        self.rate = rate
        self.chunksize = chunksize
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunksize,
                                  stream_callback=self.new_frame)
        self.lock = threading.Lock()
        self.stop = False
        self.frames = []
        atexit.register(self.close)

    def new_frame(self, data, frame_count, time_info, status):
        data = np.fromstring(data, 'int16')
        with self.lock:
            self.frames.append(data)
            if self.stop:
                return None, pyaudio.paComplete
        return None, pyaudio.paContinue

    def get_frames(self):
        with self.lock:
            frames = self.frames
            self.frames = []
            return frames

    def start(self):
        self.stream.start_stream()

    def close(self):
        with self.lock:
            self.stop = True
        self.stream.close()
        self.p.terminate()

class Application(QtGui.QMainWindow):
    def __init__(self):
        super(Application,self).__init__()
        #Window Parameters
        self.setGeometry(0,0,1366,768)
        self.setWindowTitle("HERS Configuration Utility")
        self.setWindowIcon(QtGui.QIcon('./icons/title_icon.png'))
        self.statusBar()
        mainMenu=self.menuBar()

        #Quit
        exitMenu=QtGui.QAction('&Quit',self)
        exitMenu.setShortcut('Ctrl+Q')
        exitMenu.setStatusTip('Exit the configuration Utitlity')
        exitMenu.triggered.connect(self.quit)
        fileMenu=mainMenu.addMenu('&File')
        fileMenu.addAction(exitMenu)

        imgConfig=QtGui.QAction('&Configure Images',self)
        imgConfig.setShortcut('Ctrl+I')
        imgConfig.setStatusTip('Configure Image Database')
        imgConfig.triggered.connect(self.callImageWindow)

        audConfig=QtGui.QAction('&Configure Audio',self)
        audConfig.setShortcut('Ctrl+A')
        audConfig.setStatusTip('Configure Audio DataBase')
        audConfig.triggered.connect(self.callAudioWindow)

        pulseConfig=QtGui.QAction('&Configure HeartBeat',self)
        pulseConfig.setStatusTip('Congfigure HeartBeat(Pulse) DataBase')
        pulseConfig.setShortcut('Ctrl+H')
        pulseConfig.triggered.connect(self.callPulseWindow)
        editMenu=mainMenu.addMenu('&Edit')
        editMenu.addAction(imgConfig)
        editMenu.addAction(audConfig)
        editMenu.addAction(pulseConfig)
        self.displayWindow()
        self.show()

    def initCam(self):
        #Creating Camera Object
        self.emo=detectfacial.emoDetect()
        self.cam=cv2.VideoCapture(0)
    def getEmotion(self):
        global emotion
        emotion=self.emo.predict(self.img)
    def getImage(self):
        ret,self.img=self.cam.read()
        self.img=cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        img=QtGui.QImage(self.img, self.img.shape[1], self.img.shape[0], QtGui.QImage.Format_RGB888)
        self.imgPix=QtGui.QPixmap.fromImage(img)
        self.imageLabel.setPixmap(self.imgPix)


    def callAudioWindow(self):
        global Trigger
        Trigger=2
        self.Auth()

    def callPulseWindow(self):
        print

    def callImageWindow(self):
        global Trigger
        Trigger=1
        self.Auth()

    def quit(self):
        sys.exit()

    def Auth(self):
        self.Window=Login()
        self.Window.setGeometry(300,200,300,100)
        self.Window.show()

    def displayWindow(self):
        labelHappy=QtGui.QLabel('Happy:',self)
        labelHappy.move(50,100)
        self.scaleHappy=QtGui.QProgressBar(self)
        self.scaleHappy.setGeometry(150,100,200,30)

        labelSadness=QtGui.QLabel("Sad:",self)
        labelSadness.move(50,140)
        self.scaleSadness=QtGui.QProgressBar(self)
        self.scaleSadness.setGeometry(150,140,200,30)

        labelAnger=QtGui.QLabel("Angry:",self)
        labelAnger.move(50,180)
        self.scaleAnger=QtGui.QProgressBar(self)
        self.scaleAnger.setGeometry(150,180,200,30)

        labelFear=QtGui.QLabel("Scared:",self)
        labelFear.move(50,220)
        self.scaleFear=QtGui.QProgressBar(self)
        self.scaleFear.setGeometry(150,220,200,30)

        labelSurprise=QtGui.QLabel("Surprised:",self)
        labelSurprise.move(50,260)
        self.scaleSurprise=QtGui.QProgressBar(self)
        self.scaleSurprise.setGeometry(150,260,200,30)

        self.imageLabel=QtGui.QLabel(self)
        self.imageLabel.setGeometry(720,10,640,480)
        self.imageLabel.setStyleSheet('border: 5px solid black;')
        startButton=QtGui.QPushButton('Start',self)
        startButton.move(940,500)
        startButton.clicked.connect(self.startProcess)
        stopButton=QtGui.QPushButton('Stop',self)
        stopButton.move(1040,500)
        stopButton.clicked.connect(self.stopProcess)
        '''
        #Heart Beat Display
        labelPulse=QtGui.QLabel('Pulse Rate',self)
        labelPulse.move(160,375)
        LCD=QtGui.QLCDNumber(self)
        LCD.setSegmentStyle(QtGui.QLCDNumber.Flat)
        LCD.setGeometry(130,400,120,70)
        LCD.display(8888)
        '''
        #Audio Plotting
        # mpl figure
        self.main_figure = MplFigure(self)
        self.plot_widget = QtGui.QWidget(self)
        self.plot_widget.move(0,520)
        self.plot_widget.resize(1366,180)
        plot_layout = QtGui.QVBoxLayout(self.plot_widget)
        plot_layout.addWidget(self.main_figure.canvas)
        # init class data
        self.initData()
        # connect slots
        self.connectSlots()
        # init MPL widget
        self.initMplWidget()

    def handleNewData(self):
        """ handles the asynchroneously collected sound chunks """
        # gets the latest frames
        frames = self.mic.get_frames()

        if len(frames) > 0:
            # keeps only the last frame
            current_frame = frames[-1]
            # plots the time signal
            self.line_top.set_data(self.time_vect, current_frame)

            # refreshes the plots
            self.main_figure.canvas.draw()

    def initData(self):
        mic = MicrophoneRecorder()
        mic.start()
        # keeps reference to mic
        self.mic = mic
        # computes the parameters that will be used during plotting
        self.freq_vect = np.fft.rfftfreq(mic.chunksize,
                                         1./mic.rate)
        self.time_vect = np.arange(mic.chunksize, dtype=np.float32) / mic.rate * 1000

    def connectSlots(self):
        pass

    def initMplWidget(self):
        """creates initial matplotlib plots in the main window and keeps
        references for further use"""
        self.ax_top = self.main_figure.figure.add_subplot(211)
        self.ax_top.set_ylim(-32768, 32768)
        self.ax_top.set_xlim(0, self.time_vect.max())
        # line objects
        self.line_top, = self.ax_top.plot(self.time_vect,
                                         np.ones_like(self.time_vect))
    def startProcess(self):
        self.imgCapture()

    def stopProcess(self):
        self.timer.stop()
        self.imageLabel.clear()
        del(self.cam)

    def updateEmotion(self):
        if emotion[0]!=100:
            emotion[0]+=1
        elif emotion[1]!=100:
            emotion[1]+=1
        elif emotion[2]!=100:
            emotion[2]+=1
        elif emotion[3]!=100:
            emotion[3]+=1
        elif emotion[4]!=100:
            emotion[4]+=1
        #emotion[2]=75
        #emotion[3]=25
        self.scaleHappy.setValue(emotion[2])
        self.scaleSadness.setValue(emotion[0])
        self.scaleAnger.setValue(emotion[1])
        self.scaleFear.setValue(emotion[4])
        self.scaleSurprise.setValue(emotion[3])

    def imgCapture(self):
        self.initCam()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.getImage)
        self.timer.timeout.connect(self.handleNewData)
        #self.timer.timeout.connect(self.getEmotion)
        self.timer.start(1000./30)
        self.timer.timeout.connect(self.updateEmotion)
class MplFigure(object):
    def __init__(self, parent):
        self.figure = plt.figure(figsize=(18,4),facecolor='#EEEEEE')
        self.canvas = FigureCanvas(self.figure)

def main():
    config=QtGui.QApplication(sys.argv)
    app=Application()
    sys.exit(config.exec_())

if __name__=='__main__':
    main()
