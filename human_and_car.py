import cv2
#car image
img_file ='car.jpg'
#pre-trained car classifier
classifier_file='cars.xml'

#create opencv image
img=cv2.imread('img_file')

#create car classifier
car_tracker=cv2.CascadeClassifier(classifier_file)

#convert to grew scale(needed for haar cascade)
black_n_white=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect car
cars=car_tracker.detectMultiScale(black_n_white)


#display the image with car spotted
#cv2.imshow('car detector', img)
#dont autoclose wait for a key to be pressed
#cv2.waitKey()
print("code completed")