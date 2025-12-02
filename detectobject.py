import cv2 #open cv2 library for computer vision tasks
import imutils #import imutils for image processing functions   
cam=cv2.VideoCapture(0) #initialize video capture object to read from the default camera    

firstFrame=None #initialize variable to store the first frame for background subtraction
area =500 #set minimum area size for motion detection   

while True:
    _,img=cam.read() #read a frame from the camera  
    text="Normal" #initialize text variable to indicate no movement detected

    img=imutils.resize(img,width=500) #resize the frame to a width of 500 pixels
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert the frame to grayscale
    gussian=cv2.GaussianBlur(gray,(21,21),0) #apply Gaussian blur to reduce noise and improve motion detection  
     
    if firstFrame is None: #if this is the first frame
        firstFrame=gussian #set the first frame for background subtraction
        continue #skip to the next iteration of the loop

    imdDiff=cv2.absdiff(firstFrame,gussian) #compute the absolute difference between the current frame and the first frame
    thresh=cv2.threshold(imdDiff,25,255,cv2.THRESH_BINARY)[1] #apply thresholding to get a binary image of the differences
    thresh=cv2.dilate(thresh,None,iterations=2) #dilate the thresholded image to fill in holes      
    contours,_=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #find contours in the thresholded image
    for c in contours: #loop over the contours
        if cv2.contourArea(c)<area: #if the contour area is less than the minimum area
            continue #skip to the next contour
        (x,y,w,h)=cv2.boundingRect(c) #get the bounding box coordinates for the contour
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #draw a rectangle around the detected motion
        text="Movement Detected" #update text variable to indicate movement detected
    cv2.putText(img,"Status: {}".format(text),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2) #put status text on the frame
    cv2.imshow("Camera Feed",img) #display the frame with motion detection
    key=cv2.waitKey(1) & 0xFF #wait for a key
    if key==ord("q"): #if the 'q' key is pressed
        break #exit the loop

cam.release() #release the video capture object
cv2.destroyAllWindows() #close all OpenCV windows


