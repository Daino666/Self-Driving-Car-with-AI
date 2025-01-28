import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(image, line_paramters):
    slope , intercept = line_paramters

    y1 = image.shape[0]
    y2 = int((3/5) *y1)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    return np.array([x1,y1, x2,y2])


def averge_slope_intercept(image , lines):

    left_fit = []
    right_fit = []

    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        paramters = np.polyfit( (x1,x2), (y1, y2), 1 )
        slope = paramters[0]
        intercept = paramters[1]
        if slope <0:
            left_fit.append((slope, intercept))
        else :
            right_fit.append((slope,intercept))

    if len(left_fit) and len(right_fit):
        left_fit_avg = np.average(left_fit, axis=0)
        right_fit_avg = np.average(right_fit, axis=0)


        right_line = make_coordinates(image, right_fit_avg)
        left_line = make_coordinates(image,left_fit_avg)

        return np.array([left_line, right_line])

def canny(photo):

    gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
    '''
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

        [(200,height), (1100,height), (550,250)],

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



def display_lines(image,lines):
    
    line_photo = np.zeros_like(image)
    'make a black photo with same size of actual'


    if lines is not None:
        'each line is a two dimentional like [[300 4000 5000 200]]'
        for line in lines:
            X1,Y1,X2,Y2 = line.reshape(4)    
            'reshaping here is transforming (1, 4) shape to one dimensional'
            cv2.line(line_photo, (X1,Y1), (X2,Y2), (250,0,0) , 10)
            '''
            used to draw lines on a photo using the x and y
            line photot --> photo to draw on
            first vlaue --> first point
            second value --> second point
            250 --> BGR coloring
            10--> thickness
            '''
    
    return line_photo




if __name__ == '__main__':

    cap = cv2.VideoCapture('/home/daino/Desktop/Self-Driving-Car-with-AI/Videos/test2.mp4')
    while(cap.isOpened()):

        _, frame = cap.read() 

        canny_image = canny(frame)
        masked_photo = region_of_interest(canny_image)

        Lines = cv2.HoughLinesP(masked_photo, 2,  np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap= 5 )
        average_lines = averge_slope_intercept(frame , Lines)
    
        line_image = display_lines(frame, average_lines)


        Final_image = cv2.addWeighted(frame ,0.8, line_image,1, 1 )

        cv2.imshow('result', Final_image) 

        if cv2.waitKey(2) == ord('q'):
            break
    cap.release()

    cv2.destroyAllWindows()

        #plt.imshow(canny(photo))
        #plt.show()



    # photo = cv2.imread("/home/daino/Desktop/Self-Driving-Car-with-AI/Image/test_image.jpg")
    # lane_image = np.copy(photo)
    # '''you need to make a copy of the originial array otherwise the edits
    #     you make would affect the original Array as they both would be the same array
    # '''


    # canny_image = canny(lane_image)
    # masked_photo = region_of_interest(canny_image)

    # Lines = cv2.HoughLinesP(masked_photo, 2,  np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap= 5 )
    # average_lines = averge_slope_intercept(lane_image , Lines)
    # '''
    # This Function HoghlinesP uses the Probabalistic Hough Space to get lines connecting points with each other, 
    # and with that we can now the lanes the car is moving betweeen. 

    # it uses the equation P = X*Cos(Theta) + Y*Sin(Theta)
    # this is better than the standard line equation because it can deal with infinite slopes 

    # Masked Phot   --> the photo to take lines from canny
    # RHO           --> the number of pexels per single grid 
    # Theta         --> precision in theta
    # Threshold     --> the number of contacts between sinusoidal lines to conclude a line
    # array         --> just an empy array that the function needs (search for the usage)
    # minLineLength --> Minimum length of a single line to conduct it is a line in pexels
    # maxLineGap    --> minimum pexels between two points to conduct it cant be used connected to a single line

    # '''
    # line_image = display_lines(lane_image, average_lines)


    # Final_image = cv2.addWeighted(lane_image ,0.8, line_image,1, 1 )
    # '''
    # used to add weights of pexels on each others
    # images must be same size.
    # numbers next to image variables are intensitites 
    # of images (multiply each pexel by this number).
    # last number is an offset (1 is negligible)
    # '''
    # cv2.imshow('result', Final_image) 
    # cv2.waitKey(0)
    # '''
    # THese are for showing the array we are making into image 
    # the waitkey is important for making the show function work until pressing a key 
    # '''

    # #plt.imshow(canny(photo))
    # #plt.show()
    # '''
    # we used matplotlib is it gives us the ability to see the ordered pexels numbers, which is 
    # importnat for deciding on the region of interest.
    # '''
