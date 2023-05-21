import cv2
import numpy as np
# camera
#cap = cv2.VideoCapture("vehicle.mp4")
#cap = cv2.VideoCapture('vehicle_resized.mp4')

cap = cv2.VideoCapture('vehicle4.mp4')
min_width_rect =  50#80 #width
min_height_rect = 50 #90 #height

count_line_position = 500 #550

#Initialixe Substructor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()
detect = []
offset = 6 # allowable error
counter = 0
def center_handle(x,y,w,h):
   c1 = int(w/2)
   c2 = int(h/2)
   cx = x+c1
   cy = y+c2 
   return cx,cy



while True:
 ret,frame1 = cap.read()
 grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
 blur = cv2.GaussianBlur(grey,(3,3),5)

 #applying on ech frame
 img_sub = algo.apply(blur)
 dilation  = cv2.dilate(img_sub, np.ones((5,5)))
 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
 dilatdada = cv2.morphologyEx(dilation,cv2.MORPH_CLOSE,kernel)
 dilatdada = cv2.morphologyEx(dilatdada,cv2.MORPH_CLOSE,kernel)
 counterShape,h = cv2.findContours(dilatdada,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

 cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,0,0),3)

 for(i,c) in enumerate(counterShape):
  (x,y,w,h) = cv2.boundingRect(c)
  validate_counter = (w>= min_width_rect) and (h>= min_height_rect)
  if not validate_counter:
   continue
  
  cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
  cv2.putText(frame1,"vehicle " +str(counter),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(200,100,0),2)

  center = center_handle(x,y,w,h)
  detect.append(center)
  cv2.circle(frame1,center,4,(0,0,255),-1)

  for (cx,cy) in detect:
     if cy<(count_line_position + offset) and cy>(count_line_position - offset):
        counter +=1
        crop_img = frame1[y - 5:y + h + 5, x - 5:x + w + 5]
        n = r'vehicle'+str(counter) 
        file = r'C:\Users\dangc\OneDrive\Pictures\test_vehicle' + '\\' + n + '.jpg'
        cv2.imwrite(file, crop_img)
     cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(0,200,250),3)
     detect.remove((cx,cy))
     print('vehicle counter: '+ str(counter))

 cv2.putText(frame1,"COUNTER: " +str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)

 cv2.imshow('Video Original', frame1)
 #cv2.imshow('Detector',dilatdada)
 if cv2.waitKey(5) & 0xFF == ord('q'):
        break
 



cv2.destroyAllWindows()
cap.release()








