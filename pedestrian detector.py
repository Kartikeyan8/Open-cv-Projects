#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy


# In[2]:


body_classifier=cv2.CascadeClassifier("haarcascade_fullbody.xml")   
capture=cv2.VideoCapture("walking.avi")
while capture.isOpened():
    ret,frame=capture.read()
    body=body_classifier.detectMultiScale(frame,1.1,3)
    for (x,y,w,h) in body:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('pedestrian Body detector',frame)
        if(cv2.waitKey(1) & 0xFF==ord('q')):
            break
            
    
capture.release()
cv2.destroyAllWindows()  


# In[ ]:





# In[ ]:




  


# In[49]:





# In[56]:





# In[ ]:





# In[ ]:




