import cv2

#car image
#img_file ='car.jpg'

#video on which we are working
video=cv2.videoCapture('')


#pre-trained car classifier andhuman classifier
classifier_file_cars='cars.xml'
classifier_file_humans='haarscascade_fullbody.xml'

#create car classifier
car_tracker=cv2.CascadeClassifier(classifier_file_cars)
human_tracker=cv2.CascadeClassifier(classifier_file_humans)
  
  #run until video stops
while True :

    #create opencv image
    #img=cv2.imread('img_file')
    
    #read current frame
    (read_successful,frame)=video.read() 


    #safe coding
    if read_successful:
        #covery to grayscale
        grayscaled_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break    
 
    #detect car and humans
    cars=car_tracker.detectMultiScale(grayscaled_frame)    
    humans=human_tracker.detectMultiscale(grayscaled_frame)
   
     #draw rectangle around the cars
    for(x,y,w,h)in cars:
        cv2.rectangle(frame, (x,y) ,(x+w,y+h),(0,0,255),2)
    #draw rectangle around humans
    for(x,y,w,h)in humans:
        cv2.rectangle(frame, (x,y) ,(x+w,y+h),(0,255,255),2)
        
    #display the image with car spotted
    cv2.imshow('car detector', frame)
    #dont autoclose wait for a key to be pressed
    key=cv2.waitKey(1)
    
    #stop if Q is pressed
    if key==81 or  key ==113 :
        break   
#relaease video capture
video.release()

