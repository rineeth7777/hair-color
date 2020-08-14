import numpy as np
from PIL import Image
a=np.array([[1,2],[3,4]])
b=np.array([2,3,4,5])
a=a>2
a=a.astype('uint8')
print(a)
b=np.full((2,2), 3)
b=np.random.rand(10,15,3)
#c=np.random.rand(10,)
b=np.average(b,axis=(0,1))
print('b avg',b.shape,b)

size=256
image=Image.open('D:\\AI\\vision\\data\\masks\\0.1_image\\113.jpg')
image=image.resize((size, size), Image.BILINEAR)
imarr=np.asarray(image)

image2=Image.open('D:\\AI\\vision\\data\\masks\\0.1_annotation\\00113_hair.png')
imarr2=np.asarray(image2)
imarr2=imarr2>254
imarr2=imarr2.astype('int32')

avgs=np.average(imarr,weights=imarr2,axis=(0,1))

img_w, img_h = 256, 256
data1 = np.full((img_h, img_w), avgs[0])
data2 = np.full((img_h, img_w), avgs[1])
data3 = np.full((img_h, img_w), avgs[2])
data=np.stack((data1,data2,data3),axis=2)
data=data.astype('uint8')
img = Image.fromarray(data,mode='RGB')

img.show()

x=np.array([1,2,3])
y=np.array([5,6,7])
z=np.stack((x,y),axis=1)
print('z',z)
print(z.shape)
