import cv2
from cv2 import BORDER_WRAP
from cv2 import BORDER_CONSTANT
import cv2.aruco as aruco
import numpy as np
import math

L=['D:\opencv\Atulya_Opencv\Ha.jpg','D:\opencv\Atulya_Opencv\HaHa.jpg','D:\opencv\Atulya_Opencv\LMAO.jpg','D:\opencv\Atulya_Opencv\XD.jpg']
iddict = {}
base= cv2.imread("D:\opencv\Atulya_Opencv\CVtask.jpg")
final=cv2.resize(base,(1754,1240))
def arucoid(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_5X5_250')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    (corners , ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict, parameters = arucoParam)
    return (corners , ids, rejected)
def arucocorner(img):
    (c,i,r)=arucoid(img)
    if len(c)>0:
        i=i.flatten()
        for (markercorner,markerid) in zip(c,i):
            corner = markercorner.reshape((4,2))
            (topleft,topright,bottomright,bottomleft)=corner
            topleft = (int(topleft[0]),int(topleft[1]))
            topright = (int(topright[0]),int(topright[1]))
            bottomleft = (int(bottomleft[0]),int(bottomleft[1]))
            bottomright = (int(bottomright[0]),int(bottomright[1]))
        return topleft,topright,bottomright,bottomleft

def arucoangle(img):
    topleft,topright,bottomright,bottomleft=arucocorner(img)
    cx = int((topleft[0]+bottomright[0])/2)
    cy = int((topleft[1]+bottomright[1])/2)
    px = int((topright[0]+bottomright[0])/2)
    py = int((topright[1]+bottomright[1])/2)
    m=(py-cy)/(px-cx)
    theta =math.degrees(math.atan(m))
    center = (cx,cy)
    cv2.circle(img,topright,5,(0,255,0),-1)
    cv2.circle(img,bottomright,5,(255,0,0),-1)
    cv2.circle(img,(0,0),5,(0,0,255),-1)
    return center,theta
def rotate_image(image, angle,center):
    rot_mat = cv2.getRotationMatrix2D(center, angle,0.8)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderMode=BORDER_CONSTANT,borderValue=(255,255,255))
    return result
def crp(img):
    topleft,topright,bottomright,bottomleft=arucocorner(img)
    l=[topleft,topright,bottomright,bottomleft]
    xmax=l[0][0]
    xmin=l[0][0]
    ymax=l[0][1]
    ymin=l[0][1]
    for i in l:
        if i[0]>xmax:
            xmax = i[0]
        if i[0]<xmin:
            xmin = i[0]
        if i[1]>ymax:
            ymax = i[1]
        if i[1]<ymin:
            ymin = i[1]
    print(xmax,xmin,ymax,ymin)
    test = img[ymin:ymax,xmin:xmax]
    return test
for i in L:
    x = cv2.imread(i)
    (c,ids,r)=arucoid(x)
    iddict[i]=ids
img = cv2.imread('D:\opencv\Atulya_Opencv\CVtask.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray,230,255,cv2.THRESH_BINARY)
color={'green':[79,209,146],'orange':[9,127,240],'white':[210,222,228],'black':[0,0,0]}
cid = {'green':1,'orange':2,'white':4,'black':3}
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for cont in contours:
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.01 * peri, True)

    if len(approx) == 4:
        
        x,y,w,h=cv2.boundingRect(approx)
        aspectratio = float(w)/h
        if aspectratio >=0.95 and aspectratio<=1.05:
            #print(approx)
            location = [o[0].tolist() for o in approx]
            xmax=location[0][0]
            xmin=location[0][0]
            ymax=location[0][1]
            ymin=location[0][1]
            for i in location:
                if i[0]>xmax:
                    xmax = i[0]
                if i[0]<xmin:
                    xmin = i[0]
                if i[1]>ymax:
                    ymax = i[1]
                if i[1]<ymin:
                    ymin = i[1]
            print(xmax,xmin,ymax,ymin)
            to1 = img[ymin:ymax,xmin:xmax]
            shape1 = to1.shape
            sw=shape1[0]
            se = shape1[1]
            newshape = (se-50,sw-50)
            m1=(int((location[0][0]+location[1][0])/2),int((location[0][1]+location[1][1])/2))
            c=(int((location[0][0]+location[2][0])/2),int((location[0][1]+location[2][1])/2))
            if (m1[0]-c[0]) != 0:
                theta = math.atan((m1[1]-c[1])/(m1[0]-c[0]))
            else :
                theta = math.pi/(-2)
            print (theta*180/(math.pi))
            for i in color.keys():
                d = np.array(color[i])
                d.reshape((3,))
                if (d==img[c[1],c[0],:]).any():
                    wer = np.array(cid[i])
                    wer.reshape((1,1))
                    for j in iddict.keys():
                        if (wer==iddict[j]).any():
                            sr = j
                    ar = cv2.imread(sr)
                    c1,theta1=arucoangle(ar)
                    f=rotate_image(ar,theta1-(theta*180/math.pi),c1)
                    df = crp(f)
                    s=cv2.resize(df,newshape)
                    print(shape1,s.shape)
                    final[(ymin+25):(ymax-25),(xmin+25):(xmax-25),:]=s
                    cv2.imshow('ddd',final)
                    cv2.waitKey(0)
                    cv2.putText(img,sr,c,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
            cv2.circle(final,location[0],5,(0,0,255),-1)
            cv2.drawContours(final,[approx], -1, (255, 0, 0), 3)
            print(location[0][0]-location[1][0])
