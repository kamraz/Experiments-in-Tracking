import numpy as np
import cv2 as cv
import argparse

#Takes cmd line input and opens video
parser = argparse.ArgumentParser(description='Detect and Optical Flow')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()
cap = cv.VideoCapture(args.image)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 10,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))

# Skips first 2 frames just incase of fade in or other video edits
cap.read()
cap.read()
ret, old_frame = cap.read()

#Opens GUI to select reigon of intrest
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
height, width = old_gray.shape
pts = cv.selectROI(old_gray)
print(pts)
print(len(pts))
mask = np.zeros(shape = old_gray.shape, dtype = "uint8")
cv.rectangle(img = mask, pt1 = (pts[0], pts[1]), pt2 = (pts[0]+pts[2], pts[1]+pts[3]),
	color = (255, 255, 255), thickness = -1)


#Finds points to be tracked in the ROI
p0 = cv.goodFeaturesToTrack(old_gray,4,.01,5, mask=mask)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

#Loops runs video while tracking points
while(1):
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select new points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)
    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
