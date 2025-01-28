import numpy as np 
import matplotlib.pyplot as plt


def draw(x1, x2):
    ln = plt.plot(x1,x2,'-')
    plt.pause(.0001)
    ln[0].remove()



def sigmoid(score):
    return (1/(1+np.exp(-score)))

def calc_error(Probability, Label):
    n = len(Probability)

    cross_entropy = (-1/n)*(np.log(Probability.T)*Label   + np.log(1-Probability.T)*(1-Label))
    
    return cross_entropy


def gradient_descent(points, label, line_parameters, learning_rate):

    n = points.shape[0]
    for i in range(20000):
        p = sigmoid(points*line_parameters)
        gradient = (points.T * ( p - label)) * (learning_rate/n)
        line_parameters -= gradient

        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)

        x1=np.array([points[:,0].min(), points[:,0].max()])
        x2= -b/w2 + (x1*(-w1/w2))
    
        draw(x1,x2) 

    

no_pts = 100

b = np.ones(no_pts)
x1_group1_values = np.random.normal(10, 2, no_pts)
x2_group1_values = np.random.normal(12, 3, no_pts)
upper_region = np.array([x1_group1_values, x2_group1_values, b]).T

x1_group2_values = np.random.normal(4, 2, no_pts)
x2_group2_values = np.random.normal(5, 3, no_pts)
lower_region = np.array([x1_group2_values,x2_group2_values, b]).T




all_points = np.vstack((upper_region,lower_region))

line_parameters = np.matrix(np.zeros(3)).T

score =all_points* line_parameters
probabilites = sigmoid(score)


labels = np.array([np.ones(no_pts), np.zeros(no_pts)]).reshape(no_pts*2,1)



_, ax = plt.subplots(figsize=(4,4))
ax.scatter(upper_region[:,0], upper_region[:, 1] , color = 'r')
ax.scatter(lower_region[:,0], lower_region[:, 1], color ='g')
gradient_descent(all_points, labels, line_parameters, .006)
plt.show()
#the word "color" is essential for the function to work

