import cv2
import numpy as np

# web camera 
cap=cv2.VideoCapture('vehicale_vidio.mp4')

min_width_rect=80   #minium width rectangle
min_hight_rect=80   #minimum hight rectangle
count_line_position=550
# intialize Subtractor
algo=cv2.createBackgroundSubtractorMOG2()

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy
    
detect = []
offset=6 #allowable error between pixel 
counter=0

while True:
    ret,frame1=cap.read()
    grey=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) 
    blur=cv2.GaussianBlur(grey,(3,3),5)
    # applying on  each frame
    img_sub = algo.apply(blur)
    dilat=cv2.dilate(img_sub,np.ones((5,5)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada=cv2.morphologyEx(dilat ,cv2.MORPH_CLOSE,kernel)
    dilatada=cv2.morphologyEx(dilatada ,cv2.MORPH_CLOSE,kernel)
    countershape,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #-------img---pointpass(startpoint P1)---(endpint P2)-----------,colorcode----thickness
    cv2.line(frame1,(25,count_line_position),(1400,count_line_position),(255,0,0),5)

    for (i,c) in enumerate(countershape):
        (x,y,w,h)=cv2.boundingRect(c)
        validate_counter=(w>= min_width_rect) and (h>=min_hight_rect)
        if not validate_counter:
            continue

        #------------img----pt1-----pt2-----colorcode--thickness
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,0,225),3)

        center=center_handle(x,y,w,h)
        detect.append(center)
        # ---------img,center,radius,color,thickness,linetype--
        cv2.circle(frame1,center,4,(255,255,224),-1)



        for (x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter+=1

            cv2.line(frame1,(25,count_line_position),(1400,count_line_position),(0,127,255),5)
            detect.remove((x,y))
            print("vehicle Counter:"+str(counter))


    cv2.putText(frame1,"VEHICALE COUNTER:"+str(counter),(200,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)


    #cv2.imshow('Detector',dilatada)==== this line for background data detection
    cv2.imshow("Vidio Original",frame1)
    

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()
