import cv2
import numpy as np
from os import listdir
classifiers=listdir('../data/svm/')
emotion=[0,0,0,0,0]
class GaborBank:
    #/* Gabor Formula consts */

    # Minimum bandwidth for a gabor filter
    kGaborBandwidthMin = 1.0
    #Maximum bandwidth for a gabor filter
    kGaborBandwidthMax = 2.5
    #Minimum wavelength for a gabor filter
    kGaborLambdaMin = 4
    #Maximum wavelength for a gabor filter
    kGaborLambdaMax = 16
    #Minimum sigma for a gabor filter
    kGaborSigmaMin = 2.0
    #Maximum sigma for a gabor filter
    kGaborSigmaMax = 4.0
    #Minimum orientation for a gabor filter
    kGaborThetaMin = np.pi/2.0
    #Maximum orientation for a gabor filter
    kGaborThetaMax = np.pi+np.pi/2.0

    #/* Gabor Misc consts*/

    #Gabor support shape parameter (0.5 ellipse .. 1 circle)
    kGaborGamma= 0.5
    #Gabor phase offset
    kGaborPsi=0.0

    #/* Gabor Empiric consts */

    #Empirical value for Sigma
    kGaborSigma= 1.5# 3.0
    #Empirical minimum value of wavelength
    kGaborELambdaMin = 8.0
    #Empirical maximum value of wavelength
    kGaborELambdaMax = 16.0
    #Minimum sigma for a gabor filter
    kGaborESigmaMin = 2.0
    #Maximum sigma for a gabor filter
    kGaborESigmaMax = 4.0
    #Minimum width for a gabor filter
    kGaborWidthMin = 13
    #Maximum width for a gabor filter
    kGaborWidthMax = 21


    #Default gabor number of different width (gaborbank_getGaborBank)
    kGaborDefaultNwidth = 1.0
    #Default gabor number of different lambda (gaborbank_getGaborBank)
    kGaborDefaultNlambda = 5.0
    #Default gabor number of different theta (gaborbank_getGaborBank)
    kGaborDefaultNtheta = 8.0
    #Default feature size
    kGaborDefaultFeatureSize = 40
    lastFeatureSize=kGaborDefaultFeatureSize

    def fillGaborBankEmpiric(self,nwidths,nlambdas,nthetas):
        self.bank=[]
        LAMBDAS=[0.307692,0.615385,1.23077,2.46154,4.92308]
        self.lastNtheta=nthetas
        self.lastNlambdas=nlambdas
        self.lastNwidth=nwidths
        _gamma=self.kGaborGamma
        _sigma=self.kGaborSigma
        _psi=self.kGaborPsi
        minfwidth=self.kGaborWidthMin
        maxfwidth=self.kGaborWidthMax
        _theta_step=(self.kGaborThetaMax-self.kGaborThetaMin)/nthetas
        fwidth=minfwidth
        while fwidth<maxfwidth:
            kernelsize=(fwidth,fwidth)
            for _lambda in LAMBDAS:
                _theta=self.kGaborThetaMin
                _theta_c=0
                while _theta_c<nthetas:
                    kern=self.buildGaborKernel(kernelsize,_sigma,_theta,_lambda,_gamma,_psi,cv2.CV_32F)
                    self.bank.append(kern)
                    _theta+=_theta_step
                    _theta_c+=1
            fwidth+=((maxfwidth-minfwidth)/nwidths)

    def buildGaborKernel(self,ksize,sigma,theta,lambd,gamma,psi,ktype):
        sigma_x = sigma
        sigma_y = sigma/gamma
        c = np.cos(theta)
        s = np.sin(theta)
        xmax=int(ksize[0]/2.0)
        ymax=int(ksize[1]/2.0)
        xmin=-xmax
        ymin=-ymax
        kernel_real=np.zeros(((ymax-ymin+1),(xmax-xmin+1)),np.float32)
        kernel_img=np.zeros(((ymax-ymin+1),(xmax-xmin+1)),np.float32)
        scale=1
        ex=-0.5/(pow(sigma_x,2))
        ey=-0.5/pow(sigma_y,2)
        cscale=np.pi*2/lambd
        for y in range(ymin,ymax+1,1):
            for x in range(xmin,xmax+1,1):
                #print x,y
                xr=x*c+y*s
                yr=-x*s+y*c
                v_real=scale*np.exp(ex*pow(xr,2)+ey*pow(yr,2))*np.cos(cscale*xr+psi)
                v_img=scale*np.exp(ex*pow(xr,2)+ey*pow(yr,2))*np.sin(cscale*xr+psi)
                kernel_real[ymax-y,xmax-x]=v_real
                kernel_img[ymax-y,xmax-x]=v_img
        #print kernel_real
        #kernel_real_show=cv2.resize(kernel_real,(200,200))
        #cv2.imshow('real',kernel_real_show)
        #kernel_img_show=cv2.resize(kernel_img,(200,200))
        #cv2.imshow('imag',kernel_img_show)
        return (kernel_real,kernel_img)

    def getFilteredImgSize(self,size):
        s=[0,0]
        s[1]=size[1]*40
        s[0]=size[0]
        s=tuple(s)
        return s

    def filterImage(self,src,featSize):
        bankSize=self.getFilteredImgSize(featSize)
        dest=np.zeros((bankSize[1],bankSize[0]),np.float32)
        image=np.float32(src)
        image=cv2.resize(image,featSize,interpolation=cv2.INTER_AREA)
        for k in range(0,40):
            gk=self.bank[k]
            real=gk[0]
            freal=cv2.filter2D(image,cv2.CV_32F,real)
            imag=gk[1]
            fimag=cv2.filter2D(image,cv2.CV_32F,imag)
            #cv2.imshow('filter',freal)
            #cv2.waitKey(0)
            freal=pow(freal,2)
            fimag=pow(fimag,2)
            magn=np.add(fimag,freal)
            np.sqrt(magn,magn)
            for i in range(0,featSize[1]):
                for j in range(0,featSize[0]):
                    dest[i+k*featSize[1],j]=magn[i,j]
        _min,_max,minloc,maxloc= cv2.minMaxLoc(dest)
        adj_map=cv2.convertScaleAbs(dest,255/_max)
        return adj_map

class emoDetect():
    def detectFace(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faceCascade = cv2.CascadeClassifier('./../data/haarcascade_frontalface_cbcl1.xml')
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        if (len(faces) != 1):
            return False, image
        for (x, y, w, h) in faces:
            image = image[y:y + h, x:x + w]
        cv2.imwrite('./../data/outfile.png',image)
        return True, image

    def applyFilter(self, image):
        ret, img = self.detectFace(image)
        img.astype(np.float32)
        cv2.resize(img,(52,52))
        kernel=GaborBank()
        kernel.fillGaborBankEmpiric(1,5,8)
        result=kernel.filterImage(img,(52,52))
        return result

    def predict(self,image):
        result = self.applyFilter(image)
        cv2.imshow("filter",result)
        #cv2.imshow('filter Rotated',np.rot90(result))
        cv2.waitKey(0)
        cv2.imwrite('./../data/file.png',result)
        data=result.ravel()
        for i in range(0,5):
            svm_file='./../data/svm/'+classifiers[i]
            print classifiers[i]
            svm=cv2.ml.SVM_load(svm_file)
            testdata=[]
            testdata.append(data)
            testdata=np.asarray(testdata,dtype=np.float32)
            x,ret= svm.predict(testdata,np.asarray([0],dtype=np.float32))
            print ret
            if(ret==[1.]):
                emotion[i]=1
        return emotion

if __name__=='__main__':
    emo=emoDetect()
    img=cv2.imread('./../data/testImages/happy.png')
    #img=cv2.imread('brad.jpg')
    emotion=emo.predict(img)
    print emotion