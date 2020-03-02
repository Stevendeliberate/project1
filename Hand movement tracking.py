from cv2 import cv2
import numpy as np 
from collections import deque   #双端序列 在序列的前后都可以执行添加或者删除操作

def on_EVENT_LBUTTONDOWN(event , x, y , flags , param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(hsv[y,x])

cap = cv2.VideoCapture(0)       #打开默认摄像头

pts = deque(maxlen=64)

low_hand_color = np.array([0,100,200])
Upper_hand_color = np.array([50,220,255])

while True:
    ret , img = cap.read()      #读取摄像头传入的每一帧图像到img中 是三维矩阵
                                #ret是布尔值，如果读取帧是正确的则返回True ， 如果文件读取到结尾，她的值就返回为False
    
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)   #转换格式为hsv

    cv2.imshow("hsv",hsv)
    cv2.setMouseCallback('hsv',on_EVENT_LBUTTONDOWN)
    
    kernel = np.ones((5,5),np.uint8)

    mask = cv2.inRange(hsv , low_hand_color,Upper_hand_color)   #二值化图像，阈值为之前定义的hand_color

    mask = cv2.erode(mask, kernel , iterations=1)               #腐蚀二值化的图像

    mask = cv2.morphologyEx(mask , cv2.MORPH_OPEN,kernel)       #对图像进行开运算 其实就是先腐蚀后膨胀的过程。
                                                                #开运算可以用来消除小物体，在纤细处分离物体，并且在平滑较大物体的边界的同时不明显改变其面积。
    
    mask = cv2.dilate(mask , kernel , iterations=1)             #闭运算

    res = cv2.bitwise_and(img , img , mask = mask)              #bitwise_and是对二进制数据进行“与”操作，
                                                                #即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“与”操作
                                                                #在mask上对img 和 img 进行“与”操作，即提取原图像中mask的部分

    cnts , heir = cv2.findContours(mask.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)        #轮廓提取 两个返回值contours, hierarchy
                                                                                                     #轮廓的检索模式：CV_RETR_EXTERNAL：只检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略
                                                                                                     #轮廓的近似方法：仅保存轮廓的拐点信息，把所有轮廓拐点处的点保存入contours向量内，拐点与拐点之间直线段上的信息点不予保留
    center = None

    if len(cnts) > 0:                #找到轮廓
        c = max(cnts , key = cv2.contourArea)   #在提取的轮廓中，找面积最大的轮廓
        ((x,y),radius) = cv2.minEnclosingCircle(c)  #对找到的最大的轮廓，绘制最小外切圆 ，参数为圆心和半径
        M = cv2.moments(c)          #提取最大轮廓的距 ， 找到其质心
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))       #标记质心为cente

        if radius > 5:
            cv2.circle(img , (int(x),int(y)),int(radius),(0,255,255),2) #绘制外切圆
            cv2.circle(img,center,5,(0,0,255),-1)       #绘制质点（实心圆，即为点）
        
        pts.appendleft(center)      #在pts左端添加center数据
        
        for i in range(1,len(pts)): #将得到的质点连起来，粗细由两个质点的距离决定
            if pts[i -1]is None or pts[i] is None:
                continue
            thick = int(np.sqrt(len(pts) / float( i + 1)) * 2.5)
            cv2.line(img,pts[i -1],pts[i],(0,0,255),thick)
    
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

    cv2.imshow("Frame",img)
    cv2.imshow("mask",mask)
    cv2.imshow("res",res)

cap.release()
cv2.destroyAllWindows()