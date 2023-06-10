from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog, QMessageBox
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QIcon
import math
import sys
import os
from math import log10, sqrt
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from PIL import Image
from dehazeNet_module.inference import image_haze_removel
import argparse
from skimage.metrics import structural_similarity
from DCP_module.DCP import dehaze_image_with_DCP_function

thePathOfHazzyImage = None
dehazed_photo = None
firstTime = True
thePathOfDehazedImage = None
thePathOfOriginalImage = None

# -*- coding: utf-8 -*-

def difference_betwen_photos_calculated_with_ssim(dehazed_image, clear_original_image):
    # Convert images to grayscale
    gray_dehazed_image = cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2GRAY)
    hray_clear_original_image = cv2.cvtColor(clear_original_image, cv2.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(gray_dehazed_image, hray_clear_original_image, full=True)

    return score

def difference_betwen_photos_calculated_with_mean_square_error(dehazed_image, clear_original_image):
    print('mean suqare errora basladi.')
    err = np.sum((dehazed_image.astype("float") - clear_original_image.astype("float")) ** 2)
    err /= float(dehazed_image.shape[0] * dehazed_image.shape[1])
    print('mean suqare erroru bitirdi.')
    return err


def difference_betwen_photos_calculated_with_psnr(dehazed_image, clear_original_image):
    print('psnr girdi.')
    mse = np.mean((clear_original_image - dehazed_image) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    print('psnr bitirdi.')
    return psnr


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        # Load the ui file
        uic.loadUi("GUI2.ui", self)

        # Define our widgets
        self.button1 = self.findChild(QPushButton, "add_hazy_image_button")
        self.button2 = self.findChild(QPushButton, "dehaze_image_with_DCP_button")
        self.button3 = self.findChild(QPushButton, "download_dehazed_image")
        self.button5 = self.findChild(QPushButton, "add_original_image_button")
        self.button6 = self.findChild(QPushButton, "calculate_the_sucsess_rate")
        self.button7 = self.findChild(QPushButton, "dehaze_image_with_dehaze_net_button")
        # Click The Dropdown Box
        self.button1.clicked.connect(self.add_hazy_image_button_function)
        self.button2.clicked.connect(self.dehaze_image_with_DCP_button_function)
        self.button3.clicked.connect(self.download_dehazed_image_function)
        self.button5.clicked.connect(self.add_original_image_button_function)
        self.button6.clicked.connect(self.calculate_the_sucess_rate)
        self.button7.clicked.connect(self.dehaze_image_with_dehaze_net_function)
        # Show The App
        self.show()

    def dehaze_image_with_dehaze_net_function(self):

        hazy_input_image = Image.open(thePathOfHazzyImage)
        dehaze_image = image_haze_removel(hazy_input_image)
        torchvision.utils.save_image(dehaze_image, "dehazedImages/dehazedImage.png")
        
        thePathOfDehazedImage = os.getcwd() + '/dehazedImages/dehazedImage.png';
        thePathOfDehazedImage = thePathOfDehazedImage.replace('\\', '/')

        self.pixmap = QPixmap(thePathOfDehazedImage)
        self.pixmap = self.pixmap.scaledToHeight(250)
        self.pixmap = self.pixmap.scaledToWidth(350)

        self.dehazed_image.setPixmap(self.pixmap)
    # Burası Succsess Rate Caluclation yeri.
    # Buranın da yanında bitmesi lazım
    # Bittiği zaman iş bitecek.
    # Hadi bakalım bismillah.
    def calculate_the_sucess_rate(self):

        try:
            dehazed_image= cv2.imread(os.getcwd()+"/dehazedImages/dehazedImage.png")
            clear_original_image= cv2.imread(thePathOfOriginalImage)

            mse_difference_value = difference_betwen_photos_calculated_with_mean_square_error(dehazed_image,
                                                                                          clear_original_image)
            psnr_difference_value = difference_betwen_photos_calculated_with_psnr(dehazed_image,
                                                                                          clear_original_image)
            
            ssim_difference_value = difference_betwen_photos_calculated_with_ssim(dehazed_image, clear_original_image)

            self.MSE.setText(str(mse_difference_value))
            self.PSNR.setText(str(psnr_difference_value))
            self.SSIM.setText(str(ssim_difference_value))

        except:
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Please bu sure that you upload a clear and hazzy image and deahzed it. ")
            x = msg.exec_()
            return None

    def add_hazy_image_button_function(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", "c:\\gui\\images",
                                            "All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg)")

        # Open The Image

        if fname:
            self.pixmap = QPixmap(fname[0])
            global thePathOfHazzyImage
            thePathOfHazzyImage = fname[0]
            print(thePathOfHazzyImage)
            self.pixmap = self.pixmap.scaledToHeight(250)
            self.pixmap = self.pixmap.scaledToWidth(350)

            self.hazy_image.setPixmap(self.pixmap)

    def dehaze_image_with_DCP_button_function(self):
        
        global thePathOfDehazedImage
        trasholdValue = self.the_Threshold_value.toPlainText()
        dehazedImageWithDCP = dehaze_image_with_DCP_function(thePathOfHazzyImage, trasholdValue)

        if(isinstance(dehazedImageWithDCP,int)):
            if(dehazedImageWithDCP == 1):
                print('girdi 1')
                msg = QMessageBox()
                msg.setWindowTitle("Error")
                msg.setText("Please enter a valid value. (Value betwen 0.0 and 1)")
                msg.exec_()
                return None
            elif(dehazedImageWithDCP ==2 ):
                print('girdi 2')
                msg = QMessageBox()
                msg.setWindowTitle("Error")
                msg.setText("Please Download an image to dehaze it.")
                msg.exec_()
                return None
        
        
        thePathOfDehazedImage = os.getcwd() + '/dehazedImages/dehazedImage.png';
        thePathOfDehazedImage = thePathOfDehazedImage.replace('\\', '/')
        print(dehazedImageWithDCP)
        cv2.imwrite(thePathOfDehazedImage,  dehazedImageWithDCP * 255);
        # cv2.waitKey();
        self.pixmap = QPixmap(thePathOfDehazedImage)
        self.pixmap = self.pixmap.scaledToHeight(250)
        self.pixmap = self.pixmap.scaledToWidth(350)
        # Add Pic to label

        self.dehazed_image.setPixmap(self.pixmap)
    def download_dehazed_image_function(self):

        if (thePathOfDehazedImage != None):
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                       "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)")
            if file_name:
                pixmap = QPixmap(thePathOfDehazedImage)
                pixmap.save(file_name)
                print("Image saved as:", file_name)
        else:
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Please Download an image to dehaze it.")
            x = msg.exec_()
            return None

    def add_original_image_button_function(self):
        global thePathOfOriginalImage
        fname = QFileDialog.getOpenFileName(self, "Open File", "c:\\gui\\images",
                                            "All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg)")
        print("fname: "+str(fname))
        # Open The Image

        if fname:
            thePathOfOriginalImage = fname[0] 
            originalImage = QPixmap(fname[0])
            originalImage = originalImage.scaledToHeight(250)
            originalImage = originalImage.scaledToWidth(350)

            self.original_image.setPixmap(originalImage)


# Initialize The App

app = QApplication(sys.argv)
app_icon = QIcon(os.getcwd()+"/The_Logo_1.jpg")
app.setWindowIcon(app_icon)
UIWindow = UI()
app.exec_()