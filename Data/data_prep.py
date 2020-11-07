import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os

pathIn = "./Data/target/"
pathOut = "./Data/input/"
pathColor = "./Data/color/"

names = os.listdir(pathIn)

for index in range(len(names)):
    img = cv2.imread(pathIn + names[index])
    #plt.figure()
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    #img = cv2.resize(img, (2048, 2048))
    img_edge = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    img_color = cv2.GaussianBlur(img, (101, 101), cv2.BORDER_DEFAULT)
    #img = cv2.resize(img, (512, 512))

    #cv2.imwrite("1 - denoised.png", img)
    #plt.figure()
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img_edge = cv2.Canny(img, 128, 225)
    #cv2.imwrite("2 - edge.png", img)
    #plt.figure()
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img_edge = 255 - img_edge
    #cv2.imwrite(pathOut + names[index], img_edge)
    #plt.figure()
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # img_color[img_color <= 50] = 50
    # img_color[(img_color <= 100) & (img_color > 50)] = 100
    # img_color[(img_color <= 150) & (img_color > 100)] = 150
    # img_color[(img_color <= 200) & (img_color > 150)] = 200
    # img_color[(img_color <= 255) & (img_color > 200)] = 255
    
    img_color[img_color <= 85] = 0
    img_color[(img_color <= 170) & (img_color > 85)] = 127
    img_color[(img_color <= 255) & (img_color > 170)] = 255
    
    
    cv2.imwrite(pathColor + names[index], img_color)
    
    img_color[:,:,0] = img_color[:,:,0] * (img_edge / 255)
    img_color[:,:,1] = img_color[:,:,1] * (img_edge / 255)
    img_color[:,:,2] = img_color[:,:,2] * (img_edge / 255)

    cv2.imwrite(pathOut + names[index], img_color)
    
    