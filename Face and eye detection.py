#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy


# In[ ]:


classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
input = cv2.imread("1.png")
eye_classifier=cv2.CascadeClassifier("haarcascade_eye.xml")
classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
def detect(grey,frame):
    faces=classifier.detectMultiScale(grey,3,3)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        roi_grey=grey[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        eyes=eye_classifier.detectMultiScale(roi_grey,1.1,3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),1)
    return frame
capture=cv2.VideoCapture(0)

while True:
    _,frame=capture.read()
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    eyes=eye_classifier.detectMultiScale(grey)
    canvas=detect(grey,frame)
    cv2.imshow('video',canvas)
    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break
capture.release()
cv2.destroyAllWindows()


# In[24]:





# In[42]:



  


# In[49]:





# In[56]:





# In[ ]:





# In[ ]:




