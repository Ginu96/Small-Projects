import cv2
from random import randint
import sys


font = cv2.FONT_HERSHEY_SIMPLEX
#read the vedio for tracking
vedio=cv2.VideoCapture('race.mp4')
if not vedio.isOpened():
    print("xxx")
    sys.exit()
ok,frame=vedio.read()
if not ok:
    print('error while loading')

#print(ok)
#print(frame)

#all model
all_model=['boosting','MIL','KCF','MOSSE','CSRT','TLD']
tracker=all_model[2]

if tracker=='boosting':
    tracker_model=cv2.legacy.TrackerBoosting_create()
elif tracker=='MIL':
    tracker_model=cv2.legacy.TrackerMIL_create()
elif tracker=='KCF':
    tracker_model=cv2.legacy.TrackerKCF_create()
elif tracker=='MOSSE':
    tracker_model=cv2.legacy.TrackerMOSSE_create()
elif tracker=='CSRT':
    tracker_model=cv2.legacy.TrackerCSRT_create()
elif tracker=='TLD':
    tracker_model=cv2.legacy.TrackerTLD_create()

ok,frame=vedio.read()

#select region
bbox=cv2.selectROI(frame)   

#select the intiial frame
ok=tracker_model.init(frame,bbox)

#select color for bounding box
color=(randint(0,255),randint(0,255),randint(0,255))

#
#
while True:
    ok,frame=vedio.read()
    if not ok:
        print("loading eror ")
        break
    ok,bbox=tracker_model.update(frame)
    if ok==True:
        (x,y,w,h)=[int(v) for v in bbox]
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2,1)
    else:
        cv2.putText(frame,"tracking fail",(100,90),font,.75,(0,0,255),2)
    cv2.imshow('tracking',frame)
    if cv2.waitKey(0) &0xFF==27:
        cv2.destroyAllWindows()
        break