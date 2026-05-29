---
title: ultrasound defan
date: 2022-03-25 09:10:23
tags:
- ultrasound
- image processing
categories:
- ultrasound
---

### A universal method for ultrasound image defan
![defan1](https://imgur.com/uglQwPk.jpg)


# step 1
### import moudle & load image
![load](https://imgur.com/cuaq1Cd.jpg)
```py
import cv2
from matplotlib import pyplot as plt
from util import *
from PIL import ImageDraw

img = cv2.imread(imgpath, cv2.IMREAD_COLOR)

```
&nbsp;


# step 2

### get the coner coordinate (x4)
#### here I use ginput get the coordinate

```py
plt.title('plt ginput') 
point = plt.ginput(4) 
print(point) 
```
&nbsp;


### use line1 and line2 to caculate the focus point above the img
```py
ux1, uy1 = point[0]
dx1, dy1 = point[1]
ux2, uy2 = point[2]
dx2, dy2 = point[3]
img[:int(uy1),:,:] = 0 #del area of uper UI 
line1 = [ux1, uy1, dx1, dy1]
line2 = [ux2, uy2, dx2, dy2]

focus = cross_point(line1, line2)

```
&nbsp;

### how2get cross point coordinate
```py
def cross_point(line1,line2):
    x1=line1[0]
    y1=line1[1]
    x2=line1[2]
    y2=line1[3]
    
    x3=line2[0]
    y3=line2[1]
    x4=line2[2]
    y4=line2[3]
    
    k1=(y2-y1)*1.0/(x2-x1)
    b1=y1*1.0-x1*k1*1.0
    if (x4-x3)==0:
        k2=None
        b2=0
    else:
        k2=(y4-y3)*1.0/(x4-x3)
        b2=y3*1.0-x3*k2*1.0
    if k2==None:
        x=x3
    else:
        x=(b2-b1)*1.0/(k1-k2)
    y=k1*x*1.0+b1*1.0
    return [x,y]
```
&nbsp;

# step 3
### caculating the distance of focus point to surface and bottom
![var](https://i.imgur.com/sElg0Dn.png)
```py
sur = math.dist(focus, point[0])
rad = math.dist(focus, point[1])
```
&nbsp;

### crop img and extension the matrix size
```py
img = img[:math.floor(focus[1]+rad), math.floor(dx1):math.ceil(dx2)]

mask_w = img.shape[1]
mask_h = math.ceil(rad)

bg = np.zeros(shape=(mask_h, mask_w, 3)).astype('uint8')
bg[(mask_h-img.shape[0]):mask_h,:,0] = img[:,:,0]
bg[(mask_h-img.shape[0]):mask_h,:,1] = img[:,:,1]
bg[(mask_h-img.shape[0]):mask_h,:,2] = img[:,:,2]
```
# step 4
### defan & crop img

```py
defan = de_fan(bg,probe_angle,num_aline)
defan = defan[int(sur):,:,:]
```
### de_fan code
'angle' means the probe curve angle we can get that with product info or using cosine function to get it.
'num_line' means 'A line' here we can understand that is image width. 
Here I used a for loop to resample the image, then inserted into the blank matrix.
```py
def de_fan(image,angle,num_line):
    
    h_l = math.floor(num_line/2)
    h_a = math.floor(angle/2)
    
    height, width = image.shape[:2]
    defan = np.zeros(shape=(height,num_line,3)).astype('uint8')
    defan[:,h_l,:] = image[:,math.floor(width)-1,:]
    
    for d in range(0,h_l):
        line = rotate_image_topcenter(image,d*h_a/h_l)
        defan[:,h_l-d] = line[:,int(line.shape[1]/2)]
        line = rotate_image_topcenter(image,-d*h_a/h_l)
        defan[:,h_l+d] = line[:,int(line.shape[1]/2)]
    return defan
```
&nbsp;
### rotate_image_topcenter code
Let rotate center is focused center

```py
def rotate_image_topcenter(mat, angle):
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, 0) 
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (width, height))
    return rotated_mat

```
&nbsp;
finally---
![fin](https://i.imgur.com/yPYAMkn.png)
&nbsp;

