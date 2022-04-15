# fundamental package for scientific computing with Python
import numpy as np 
# package to solve computer vision problems
import cv2
# command line argument passing module 
import argparse 
# packing for various image processing operations
import imutils 
# Parser for command line options
ap = argparse.ArgumentParser() 
# path for input video file
ap.add_argument("-v", "--video", help="path to the video file") 
# specify minimum area
ap.add_argument("-a", "--min-area", type=int, default=2500, help="minimum area size")
# parsing command line arguments  
args = vars(ap.parse_args()) 
# reading video file
vs = cv2.VideoCapture(args["video"]) 
# removing shadows
fgbg = cv2.createBackgroundSubtractorMOG2(128,cv2.THRESH_BINARY,1) 
# creating array for storing centroids
centorid = []
# creating array for storing previous centroids
prev_centroid =[]
# creating array for storing count of persons going inside
prev_in_count = 0
# creating array for storing count of persons going outside
prev_out_count = 0
#initialising initial count for in to zero
in_count = 0
#initialising initial count for out to zero
out_count = 0
#initialising total count for in to zero
totalUp = 0
#initialising total count for out to zero
totalDown = 0
# reading background image
img = cv2.imread('frame.jpg')
# converting background image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# adding blur to image
gray = cv2.GaussianBlur(gray, (21, 21), 0)
# final converted background image
firstFrame = gray
# initialising number of frames 
totalFrames = 0
while(1):
    in_count = 0
    out_count = 0
    # reading video frame by frame
    ret, frame = vs.read()
    centorid = []
    # resizing captured frame
    frame = imutils.resize(frame, width=500)
    # setting height and width of the captured frame
    (H,W) =  frame.shape[:2]
    # selected region of interest frame
    frame1 = frame[50:210, 39:199]
    # resizing region of interest frame
    frame1 = imutils.resize(frame1, width=500)
    # checking frame
    if frame is None:
        break
    if totalFrames % 1 == 0:
        # converting captured frame to grayscale
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        # adding blur to captured frame
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        # calculating absolute difference for background subtraction
        frameDelta = cv2.absdiff(firstFrame, gray)
        # setting threshold value
        thresh = cv2.threshold(frameDelta, 70, 255, cv2.THRESH_BINARY)[1]
        # showing the frame
        cv2.imshow("Asla",thresh)

        # filling the holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        # for displaying contours
        # Comment these two lines as this was added for detecting shadows
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            centre1 = []
        # if the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                continue
            # setting attributes of bounding box
            (x, y, w, h) = cv2.boundingRect(c)
            # calculating x value of centroid
            midpointX = (x + x + w) // 2
            # calculating y value of centroid
            midpointY = (y + y  ) // 2
            # appending x value of centroid
            centre1.append(midpointX)
            # appending y value of centroid
            centre1.append(midpointY)
            # appending centroid to centroid array
            centorid.append(centre1)
            #print(centorid)
            # drawing bounding box around person
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # logic for deciding in or out?
        for x in centorid:
            # if x is less than specified threshold value
            if x[1] < 140:
                # increment out count
                out_count += 1
                #print(out_count,"OUT")

            else:
                # increment in count
                in_count += 1
                #print(in_count,"IN")
        # print(out_count,in_count)
        # calculating differnce between current out count with previous one 
        op = out_count - prev_out_count
        # calculating differnce between previous in count with current one
        pi =  prev_in_count - in_count
        # calculating differnce between current in count with previous one
        ip = in_count - prev_in_count
        # calculating differnce between previous out count with current one
        po = prev_out_count - out_count
        # print(ps,sp)
        if pi > 0 and op > 0 :
            if pi == op :
                # count for total persons going in 
                totalDown += (prev_in_count - in_count)
            

        elif po > 0  and ip > 0:
            if po == ip:
                # count for total persons going out
                totalUp += po
            
    # print(totalUp)
    # print("DOWN")
    # print(totalDown)
    # information array to display counter on frame
    info = [
        ("In", totalUp),
        ("Out", totalDown),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        # putting our text on frame
        cv2.putText(frame, text, (10, H//3 - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # checking if person is still there    
    if len(centorid) != 0 :
        # storing value of cantroid if value of current centroid is not zero
        prev_centroid = centorid
    # storing in count    
    prev_in_count = in_count
    # storing out count
    prev_out_count = out_count
    # showing our video frame
    cv2.imshow('frame',frame)
    # showing only region of interest frame
    cv2.imshow("Frame", frame1)
    # incrementing frame number
    totalFrames += 1
    # wait  for 30 ms
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
# releasing video capture
cap.release()
# destroying all windows
cv2.destroyAllWindows()
