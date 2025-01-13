import cv2
import numpy as np

photo = cv2.imread("/home/daino/Desktop/Self-Drivin Car course/Image/test_image.jpg")


cp_photo = np.copy(photo)
gray = cv2.cvtColor(cp_photo, cv2.COLOR_RGB2GRAY)
'''you need to make a copy of the originial array otherwise the edits
you make would affect the original Array as they both would be the same array

Making a gray image is important for detecting edge as it make it make every pexel has one valye, hence,
reduce computation for edge detection (which is made by comparing the diffrence between pexelx values)
'''

blur = cv2.GaussianBlur(gray, (5,5), 0)
'''
Bluring the photo is important for smoothing edges and hence, reducing the noises in the photo so we can make better edge detection,
Later on using the canny function would autonmatically apply the (5,5) kernel filter for bluring 
'''

canny = cv2.Canny(blur, threshold1= 50, threshold2= 150)
'''
This function gets the derivative of pexels which each of its surrounding pexels,
if the dervative is large then it is an edge if not then it is not considred an edge.
the first variable is the photo, second is the lowest threshold, third is the highest threshold.

Below  low threshold --> dont specify an edge --> black pexel 
Higher than  high threshold --> specifiy and edge --> draw a white pexel indicating an edge
Between low and High --> specify an edge if only there is strong edges in surriunding.
'''




cv2.imshow('result', canny) 
cv2.waitKey(0)
'''
THese are for showing the array we are making into image 
the waitkey is important for making the show function work until pressing a key 
'''