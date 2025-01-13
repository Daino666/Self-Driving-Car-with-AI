import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(photo):

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

    return canny



def region_of_interest(photo):

    height = photo.shape[0]
    '''
    this gives us the number of the last pexel (in the bottom left of the photo)
    '''

    polygons = np.array([

        [(300,height), (1000,height), (560,270)],

            ])
    '''
    here we difine the boundires of polygon we want our mask to focus on
    note that the variable polygons must be an array of mutliple polygons 
    to fit in the fillpolly cv2 function, this is why we made an array of arrays in it 

    '''
    
    
    mask = np.zeros_like(photo)
    '''
    making an array of zeros having the same hight and width of our photo
    '''


    cv2.fillPoly(mask, polygons, 255)
    '''
    this line fits the polygons into the mask we made
    Masks --> the photo we want to fit polygons in
    polygons --> the figures we want to fit in the photo

    255 --> the values of the pexels (here 255 means white while it means RGB it differs)  
    '''

    masked_photo = cv2.bitwise_and(photo,mask)


    return masked_photo







photo = cv2.imread("/home/daino/Desktop/Self-Driving-Car-with-AI/Image/test_image.jpg")
canny = canny(photo)
masked_photo = region_of_interest(canny)

cv2.imshow('result', masked_photo) 
cv2.waitKey(0)
'''
THese are for showing the array we are making into image 
the waitkey is important for making the show function work until pressing a key 
'''

#plt.imshow(canny(photo))
#plt.show()
'''
we used matplotlib is it gives us the ability to see the ordered pexels numbers, which is 
importnat for deciding on the region of interest.
'''