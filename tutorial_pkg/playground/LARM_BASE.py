import cv2
import numpy as np

im0=np.zeros((10,10),dtype=int)
im0=np.uint8(im0)
im0[5,5]=255
fig0=cv2.imshow("im0",im0)
cv2.waitKey()
cv2.destroyWindow("im0")

im0=range(250)
im0=np.reshape(im0,(10,25))
im0=np.uint8(im0)
print("Valeur du pixel en coordonnées (lig=5, col=10) -> im0[5,10] : "+str(im0[5,10]))
fig0=cv2.imshow("im0",im0)
cv2.waitKey()
cv2.destroyWindow("im0")

im_gray=cv2.imread("image1.jpg",0)
cv2.imshow("Image niveaux de gris",im_gray)
print("Valeur du pixel en coordonnées (lig=50, col=50) -> im_gray[50,50] : "+str(im_gray[50,50]))
cv2.waitKey()
cv2.destroyWindow("Image niveaux de gris")

im_couleur=cv2.imread("fleur.jpg",1)
print("Valeur du pixel en coordonnées (lig=50, col=50) -> im_couleur[50,50] : "+str(im_couleur[50,50]))
b,g,r = cv2.split(im_couleur) 

rgray=cv2.merge((r,r,r))
ggray=cv2.merge((g,g,g))
bgray=cv2.merge((b,b,b))

imRGB1=rgray
imRGB1=np.concatenate((imRGB1,ggray), axis=1)
imRGB1=np.concatenate((imRGB1,bgray), axis=1)

imZeros=np.zeros(r.shape,np.uint8)
imRed=cv2.merge((imZeros,imZeros,r))
imGreen=cv2.merge((imZeros,g,imZeros))
imBlue=cv2.merge((b,imZeros,imZeros))

imRGB2=imRed
imRGB2=np.concatenate((imRGB2,imGreen), axis=1)
imRGB2=np.concatenate((imRGB2,imBlue), axis=1)

imRGB=imRGB1
imRGB=np.concatenate((imRGB,imRGB2), axis=0)

cv2.imshow("Trois canaux (R,G,B)",imRGB)
cv2.imshow("Image couleur",im_couleur)

cv2.waitKey()

















