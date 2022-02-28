import cv2
import numpy as np
from PIL import Image

img2= cv2.imread("./sample/o.jpg")

#display input slide
cv2.imshow('slide',img2)
cv2.waitKey(0)

#color plane extraction(green plane extraction)
green_channel = img2[:,:,1]

#binary conversion
ret, thresh1 = cv2.threshold(green_channel, 120, 255, cv2.THRESH_BINARY)


im_ivt=cv2.bitwise_not(thresh1)
cv2.imwrite('inverted.jpg',im_ivt)
im=cv2.imread("inverted.jpg")



#column-wise split -group A,group B,group Rh
height,width,channels=im.shape

width_cutoff = width // 3
sA = im[:, :width_cutoff]
sB = im[:, width_cutoff+1:2*width_cutoff]
sRh = im[:,int(2*width_cutoff)+1:]


#canny edge detection
eA=cv2.Canny(sA,30,300)
eB=cv2.Canny(sB,30,300)
eRh=cv2.Canny(sRh,30,300)

cv2.imshow('canny edges A',eA)
cv2.imshow('canny edges B',eB)
cv2.imshow('canny edges Rh',eRh)
cv2.waitKey(0)


#counting objects(closed edges which can represent agglutinated clump)
cA, hierarchy= cv2.findContours(eA.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cB, hierarchy= cv2.findContours(eB.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cRh, hierarchy= cv2.findContours(eRh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
nA= len(cA)
nB= len(cB)
nRh= len(cRh)
print ("The number of objects in this image: ", str(nA))
print ("The number of objects in this image: ", str(nB))
print ("The number of objects in this image: ", str(nRh))


#blood group detection(considering 32 arbitrarily refering the research doc)
nA=1 if nA>32 else 0
nB=1 if nB>32 else 0
nRh=1 if nRh>32 else 0

if nA==1 and nB==1 and nRh==1:
    print('Blood type:AB')
elif nA==1 and nB==0 and nRh==1:
    print('Blood type:A')
elif nA==0 and nB==1 and nRh==1:
    print('Blood type:B')
else:
    print('Blood type:O')
