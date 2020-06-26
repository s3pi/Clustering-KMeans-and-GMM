# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 18:03:23 2018

@author: Group02(Ganesan, Preethi)
Purpose: Program to segment the cell image data given the cluster center after k-means algorithm..

"""

#import cv2 as cv
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imsave

#class segment_img(object):
        
def segmentCell_Image(whcluster, testclsf):
    img_path = "/home/uay_user/PRAssignment2/cervical_cytology/Test/56.png"
    img_arr = imread(img_path)
    rows = img_arr.shape[0]
    cols = img_arr.shape[1]
    d3 = 3
    d1, d2 = testclsf.shape
#        rgb = np.zeros((rows,cols,d3), dtype=np.uint8)
#        patch = np.zeros((7,7,d3), dtype=np.uint8)
    rb_img_array = np.zeros((rows,cols,3), dtype=np.uint8)
    multp = rows - 6
    for i in range(rows - 6):
        for j in range(cols - 6):
            if(whcluster[i * multp + j] == 0):
                rb_img_array[i+3,j+3,0] = 255
            if(whcluster[i * multp + j] == 1):
                rb_img_array[i+3,j+3,1] = 255
            if(whcluster[i * multp + j] == 2):
                rb_img_array[i+3,j+3,2] = 255
    
    imsave('output.png', rb_img_array)
    print ("done")
    
def segmentcell(k):
    img_path = "/home/uay_user/PRAssignment2/cervical_cytology/Test_img/91.png"
    img_arr = imread(img_path)
    rows = img_arr.shape[0]
    cols = img_arr.shape[1]
#        rgb = np.zeros((rows,cols,d3), dtype=np.uint8)
#        patch = np.zeros((7,7,d3), dtype=np.uint8)
    rb_img_array = np.zeros((rows,cols,3), dtype=np.uint8)
    i, j = 0, 0
    while i < rows-6:
        while j < cols-6:
            patch = img_arr[i:i+7,j:j+7]
            mu = np.mean(patch)
            st = np.std(patch)
            X = np.array([mu,st])
            whc = assigncenter(X, k)
            if(whc == 0):
                rb_img_array[i,j,0] = 255
            if(whc == 1):
                rb_img_array[i,j,1] = 255
            if(whc == 2):
                rb_img_array[i,j,2] = 255
#            print (i, j)
            j = j+1
        i = i+1
        j = 0
    imsave('output.png', rb_img_array)
    np.save('img_np_array.npy',rb_img_array)
    print ("done")

    
def assigncenter(X, k):
    # Converged center after k-means
#    a = np.array([ 220.38408548,15.93481087,186.15172549,433.08519923,147.14655506,1464.61799681])
    a = np.array([192.1140305,5.77947889,231.17896324,3.10801129, 149.53922032,9.5996049])
#    a = np.array([221.52207203,17.97710984,229.47696225,2.36915416,228.09895031,2.5197148])
#    a = np.array([149.4313603,9.61198593,192.04776521,5.78161226,231.17464648,3.10878792])

#    b = np.array([[225.1937264    3.39686027] [235.21763707   3.09021477][171.57372072   7.36813859]])
    n,d = 1,2
    center = a.reshape(k,d) 
    distances = np.zeros((n, k))
   
#        print (center)
#        print (center[0])
    
    for i in range(k):
        distances[:,i] = np.linalg.norm(center[i] - X, axis =0)
    whcluster = np.argmin(distances, axis=1)
#        print (whcluster)
#        cluster_data = []
#        cluster_data = np.array([testclsf[whcluster==i] for i in range(k)])
    return whcluster
#

def main():
    k = 3
    d = 2
    segmentcell(k)
#
##        center = np.array([[ 220.38408548   15.93481087] 
##        [ 186.15172549  433.08519923]
##        [ 147.14655506 1464.61799681]])
main()
    

 
   
