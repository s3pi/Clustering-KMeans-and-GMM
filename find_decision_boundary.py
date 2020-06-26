# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 11:06:47 2018

@author: Group02(Ganesan, Preethi)
Purpose: After GMM convergence, the mu, pi, sigma array stored and read in the program and the decision boundary is plotted. 
"""

from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd

def Find_decision_boundary_plot(data_some_classes, pi_each_class_each_k, mu_each_class_each_k, sigma_each_class_each_k, k, btw_num_of_classes, class_names,selected_colors):
    colorUse = ("red", "green", "blue")
    catColor = (u'#FFAFAF', u'#BBFFB9', u'#BBB9FF')
    contourColor = {"Purples","Greens","Blues"}  
    groupName = ("Class1", "Class2", "Class3")
    markSym = ('o', '^', 's')
    plt.xlabel('Xaxis')
    plt.ylabel('Yaxis')
    plt.figure()
    cls_xmin,cls_xmax,cls_ymin,cls_ymax = [],[],[],[]
    for clsno in range(btw_num_of_classes):
        xval = data_some_classes[clsno][:,0]
        yval = data_some_classes[clsno][:,1]
        xmax = np.max(xval)
        ymax = np.max(yval)
        xmin = np.min(xval)
        ymin = np.min(yval)
        cls_xmin.append(xmin)
        cls_ymin.append(ymin)
        cls_xmax.append(xmax)
        cls_ymax.append(ymax)
    xmin1 = min(cls_xmin) - ((min(cls_xmax) - min(cls_xmin)) * 0.25)
    ymin1 = min(cls_ymin) - ((min(cls_ymax) - min(cls_ymin)) * 0.25)
    xmax1 = max(cls_xmax) + ((min(cls_xmax) - min(cls_xmin)) * 0.25)
    ymax1 = max(cls_ymax) + ((min(cls_ymax) - min(cls_ymin)) * 0.25)
    print (xmin1, ymin1, xmax1, ymax1)
    x0,y0,x1,y1,x2,y2 = [], [],[],[],[],[]    
    for a in np.arange(xmin1, xmax1, (xmax1-xmin1)/200.0):
        for b in np.arange(ymin1, ymax1, (ymax1-ymin1)/200.0):
           xy = [a,b]
           pt_in_space = np.array(xy)
           whclass = test_which_class(pt_in_space, pi_each_class_each_k, mu_each_class_each_k, sigma_each_class_each_k, k, btw_num_of_classes)
           if(whclass==0):
               x0.append(a)
               y0.append(b) 
           elif(whclass==1):
               x1.append(a)
               y1.append(b) 
           else: 
               x2.append(a)
               y2.append(b)
    plt.scatter(x0,y0,alpha=0.4,marker='s', edgecolors=catColor[selected_colors[0]], facecolor=catColor[selected_colors[0]], s=50)
    plt.scatter(x1,y1,alpha=0.4,marker='s', edgecolors=catColor[selected_colors[1]], facecolor=catColor[selected_colors[1]], s=50)
    plt.scatter(x2,y2,alpha=0.4,marker='s', edgecolors=catColor[selected_colors[2]], facecolor=catColor[selected_colors[2]], s=50)
    for clsno in range(0,btw_num_of_classes):
        xval = data_some_classes[clsno][:,0]
        yval = data_some_classes[clsno][:,1]
        plt.scatter(xval,yval,alpha=1.0,marker=markSym[clsno],edgecolors=colorUse[selected_colors[clsno]],facecolor=colorUse[selected_colors[clsno]],s=5,label=class_names[clsno])
    plt.legend()
    plt.xlabel('Xaxis')
    plt.ylabel('Yaxis')
#    plt.axes().set_aspect('equal', 'box')
    plt.savefig("1.png", dpi = 500)
    plt.savefig("Type"+str(class_names[0])+str(class_names[1])+str(class_names[2])+".png")
    
def test_which_class(x, pi_each_class_each_k, mu_each_class_each_k, sigma_each_class_each_k, k, btw_num_of_classes):
    prob_xi_each_class = []
    for i in range(btw_num_of_classes):
        summation = 0
        for j in range(k):
            nix = norm_pdf_multivariate(x, mu_each_class_each_k[i][j], sigma_each_class_each_k[i][j])
            summation += pi_each_class_each_k[i][j] * nix
        prob_xi_each_class.append(summation)

    return prob_xi_each_class.index(max(prob_xi_each_class))

def compute_confusion_matrix(data_all_classes_test, pi_each_class_each_k, mu_each_class_each_k, sigma_each_class_each_k, k):
    confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(len(data_all_classes_test[i])):
            class_num = test_which_class(data_all_classes_test[i][j], pi_each_class_each_k, mu_each_class_each_k, sigma_each_class_each_k, k)
            confusion_matrix[i][class_num] += 1
            print ("here")
    return confusion_matrix

def norm_pdf_multivariate(x, mu, sigma):
    #print("x", x, "mu", mu, "sigma", sigma)
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = round(np.linalg.det(sigma), 6)
#        print(det)
        if det <= 0:
            return 0
        #print(det)
        denomi = (math.pow((2*math.pi),float(size)/2) * math.pow(det,1.0/2))       
        norm_const = 1.0/ denomi
        x_mu = np.matrix(x - mu) 
        inv = np.linalg.inv(sigma)        
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")
def main():
#    data_all_classes = np.load('data_some_classes.npy')
    data_1 = pd.read_csv("/home/uay_user/PRAssignment2/cervical_cytology/Test_Feature_Vectors/51_7j.txt", header = None, delimiter=',', usecols=(0, 1)) 
    data_all_classes = np.array(data_1)
    pi_each_class_each_k = np.load('pi_each_class_each_k.npy')
    mu_each_class_each_k = np.load('mu_each_class_each_k.npy')
    sigma_each_class_each_k = np.load('sigma_each_class_each_k.npy')
    k = 3                   #Number of clusters
    btw_classes = 5         #plot contour plots between the selected classes
    dimension_of_data = 2   #please specify the dimension of data here.. 

    #arrange the information
    if (btw_classes == 1):
        data_some_classes = [data_all_classes[0], data_all_classes[1]]
        pi_some_class_each_k = [pi_each_class_each_k[0], pi_each_class_each_k[1]]
        mu_some_class_each_k = [mu_each_class_each_k[0], mu_each_class_each_k[1]]
        sigma_some_class_each_k = [sigma_each_class_each_k[0], sigma_each_class_each_k[1]]
        btw_num_of_classes = 2
        class_names = ['C1','C2','']
        selected_colors = [0, 1, 2]
    elif (btw_classes == 2):
        data_some_classes = [data_all_classes[1], data_all_classes[2]]
        pi_some_class_each_k = [pi_each_class_each_k[1], pi_each_class_each_k[2]]
        mu_some_class_each_k = [mu_each_class_each_k[1], mu_each_class_each_k[2]]
        sigma_some_class_each_k = [sigma_each_class_each_k[1], sigma_each_class_each_k[2]]
        btw_num_of_classes = 2
        class_names = ['C2','C3','']
        selected_colors = [1, 2, 0]
    elif (btw_classes == 3):
        data_some_classes = [data_all_classes[0], data_all_classes[2]]
        pi_some_class_each_k = [pi_each_class_each_k[0], pi_each_class_each_k[2]]
        mu_some_class_each_k = [mu_each_class_each_k[0], mu_each_class_each_k[2]]
        sigma_some_class_each_k = [sigma_each_class_each_k[0], sigma_each_class_each_k[2]]
        btw_num_of_classes = 2
        class_names = ['C1','C3','']
        selected_colors = [0, 2, 1]
    elif (btw_classes == 4):
        data_some_classes = data_all_classes
        pi_some_class_each_k = pi_each_class_each_k
        mu_some_class_each_k = mu_each_class_each_k
        sigma_some_class_each_k = sigma_each_class_each_k
        btw_num_of_classes = 3
        class_names = ['C1','C2','C3']
        selected_colors = [0, 1, 2]
    elif (btw_classes == 5):
        data_some_classes = data_all_classes[0]
        pi_some_class_each_k = pi_each_class_each_k[0]
        mu_some_class_each_k = mu_each_class_each_k[0]
        sigma_some_class_each_k = sigma_each_class_each_k[0]
        btw_num_of_classes = 1
        
#    Find_decision_boundary_plot(data_some_classes, pi_some_class_each_k, mu_some_class_each_k, sigma_some_class_each_k, k, btw_num_of_classes, class_names, selected_colors)
        
    print ("here")
    
main()
