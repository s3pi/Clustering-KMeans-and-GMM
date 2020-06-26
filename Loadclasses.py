# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:50:20 2018

@author: uay_user
"""
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

class loadData:
    #Loads Non-Linearly separable data
    def load_NLS_trainingset():
        data_1 = pd.read_csv("/home/uay_user/PRAssignment2/NLS/train_class1.txt", header = None, delimiter=' ', usecols=(0, 1))
        data_2 = pd.read_csv("/home/uay_user/PRAssignment2/NLS/train_class2.txt", header = None, delimiter=' ', usecols=(0, 1))
        data_3 = pd.read_csv("/home/uay_user/PRAssignment2/NLS/train_class3.txt", header = None, delimiter=' ', usecols=(0, 1))
        dataf = np.concatenate((data_1, data_2, data_3))
        trncls1 = np.array(data_1)
        trncls2 = np.array(data_2)
        trncls3 = np.array(data_3)
        trnclsf = [trncls1, trncls2, trncls3]
        
    #    prob1 = len(data_1) / len(data_1) + len(data_2) + len(data_3)
    #    prob2 = len(data_2) / len(data_1) + len(data_2) + len(data_3)
    #    prob3 = len(data_3) / len(data_1) + len(data_2) + len(data_3)
    #    prob_NLS = [prob1, prob2, prob3]
        return trnclsf
        
    def load_NLS_test_set():
        pd_data_class1 = pd.read_csv("/home/uay_user/PRAssignment2/NLS/test_class1.txt", header = None, delimiter=' ', usecols=(0, 1))
        pd_data_class2 = pd.read_csv("/home/uay_user/PRAssignment2/NLS/test_class2.txt", header = None, delimiter=' ', usecols=(0, 1))
        pd_data_class3 = pd.read_csv("/home/uay_user/PRAssignment2/NLS/test_class3.txt", header = None, delimiter=' ', usecols=(0, 1))
        #dataf = np.concatenate((pd_data_class1, pd_data_class2, pd_data_class3)) #2 dimentional - 1500 * 2
        np_data_class1 = np.array(pd_data_class1)
        np_data_class2 = np.array(pd_data_class2)
        np_data_class3 = np.array(pd_data_class3)
        data_all_classes_test = [np_data_class1, np_data_class2, np_data_class3] #3 dimentional - 3 * training_class_size * 2
        
        return data_all_classes_test
    
    #Loads ReadWorld SceneImage data
    
    #Loads Real-world speech data 
    def load_speech_trainingset(k):
        data_1 = pd.read_csv("/home/uay_user/PRAssignment2/RD/train_class1.txt", header = None, delimiter=' ', usecols=(0, 1))
        data_2 = pd.read_csv("/home/uay_user/PRAssignment2/RD/train_class2.txt", header = None, delimiter=' ', usecols=(0, 1))
        data_3 = pd.read_csv("/home/uay_user/PRAssignment2/RD/train_class3.txt", header = None, delimiter=' ', usecols=(0, 1))
        dataf = np.concatenate((data_1, data_2, data_3))
        trncls1 = np.array(data_1)
        trncls2 = np.array(data_2)
        trncls3 = np.array(data_3)
        trnclsf = [trncls1, trncls2, trncls3]
        return trnclsf
        
    def load_speech_testset(k):
        data_1 = pd.read_csv("/home/uay_user/PRAssignment2/RD/test_class1.txt", header = None, delimiter=' ', usecols=(0, 1))
        data_2 = pd.read_csv("/home/uay_user/PRAssignment2/RD/test_class2.txt", header = None, delimiter=' ', usecols=(0, 1))
        data_3 = pd.read_csv("/home/uay_user/PRAssignment2/RD/test_class3.txt", header = None, delimiter=' ', usecols=(0, 1))
        dataf = np.concatenate((data_1, data_2, data_3))
        trncls1 = np.array(data_1)
        trncls2 = np.array(data_2)
        trncls3 = np.array(data_3)
        trnclsf = [trncls1, trncls2, trncls3]
        return trnclsf
    #Loads Cervical_cytology Cell image  data
    def load_CC_trainingset():
        data_1 = pd.read_csv("/home/uay_user/PRAssignment2/cervical_cytology/Train_Feature_Vectors/train_fetr.txt", header = None, delimiter=',', usecols=(0, 1)) 
        cc_data = np.array(data_1)
        return cc_data
        
    #Loads Cervical_cytology Cell image  data
    def load_CC_testset():
        data_1 = pd.read_csv("/home/uay_user/PRAssignment2/cervical_cytology/Test_Feature_Vectors/51_7j.txt", header = None, delimiter=',', usecols=(0, 1)) 
        cc_data = np.array(data_1)
        return cc_data
        
    def load_SI_trainingset():
        train_folder = ["coast", "industrial_area", "pagoda"]    
        dir_path = "/home/uay_user/PRAssignment2/SceneImage_features/Train_Feature_Vectors"
        i = 0
        trnclsf = []
        for scene_type in train_folder:
            all_features_path = join(dir_path, scene_type)
            frame = pd.DataFrame()
            dataA = []
            for each_img_file_name in listdir(all_features_path):
                img_path = join(all_features_path, each_img_file_name)
#                print (img_path)
                cols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)
                data_1 = pd.read_csv(img_path, header = None, delimiter=',', usecols = cols)
                dataA.append(data_1)
                i = i+ len(data_1)
            frame = pd.concat(dataA)
            frame_ip = [tuple(i) for i in frame.as_matrix()]
            dataf = np.array(frame_ip)
            trnclsf.append(dataf)
        return trnclsf
    
    def load_SI_testset():
        train_folder = ["coast", "industrial_area", "pagoda"]    
        dir_path = "/home/uay_user/PRAssignment2/SceneImage_features/Test_Feature_Vectors"
        dataA = []
        i = 0
        trnclsf = []
        for scene_type in train_folder:
            all_features_path = join(dir_path, scene_type)
            frame = pd.DataFrame()
            for each_img_file_name in listdir(all_features_path):
                img_path = join(all_features_path, each_img_file_name)
#                print (img_path)
                cols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)
                data_1 = pd.read_csv(img_path, header = None, delimiter=',', usecols = cols)
                dataA.append(data_1)
                i = i+ len(data_1)
            frame = pd.concat(dataA)
            frame_ip = [tuple(i) for i in frame.as_matrix()]
            dataf = np.array(frame_ip)
            trnclsf.append(dataf)
        return trnclsf
    
    def load_Bovw_trainingset():
        train_folder = ["coast", "industrial_area", "pagoda"]    
        dir_path = "/home/uay_user/PRAssignment2/SceneImage_features/Train_BovW_Feature_Vectors"
        
        i = 0
        trnclsf = []
        for scene_type in train_folder:
            all_features_path = join(dir_path, scene_type)
            frame = pd.DataFrame()
            dataA = []
            for each_img_file_name in listdir(all_features_path):
                img_path = join(all_features_path, each_img_file_name)
#                print (img_path)
                cols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31)
                data_1 = pd.read_csv(img_path, header = None, delimiter=',', usecols = cols)
                dataA.append(data_1)
                i = i+ len(data_1)
            frame = pd.concat(dataA)
            frame_ip = [tuple(i) for i in frame.as_matrix()]
            dataf = np.array(frame_ip)
            trnclsf.append(dataf)
#            trnclsf.append(dataf)
        return trnclsf

        
#    def load_SI_trainingset():
#        train_folder = ["coast", "industrial_area", "pagoda"]    
#        dir_path = "/home/uay_user/PRAssignment2/SceneImage_features/Train_Feature_Vectors"
#        dataA = []
#        frame = pd.DataFrame()
#        i = 0
#        for scene_type in train_folder:
#            all_features_path = join(dir_path, scene_type)
#            for each_img_file_name in listdir(all_features_path):
#                img_path = join(all_features_path, each_img_file_name)
#                print (img_path)
#                cols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)
#                data_1 = pd.read_csv(img_path, header = None, delimiter=',', usecols = cols)
#                dataA.append(data_1)
#                i = i+ len(data_1)
#        frame = pd.concat(dataA)
#        dataf = frame.as_matrix()
#        return dataf        
       
#def contour_plot_some(cluster_d, k,tot_rows):
#    cat_color = (u'#FFAFAF', u'#BBFFB9', u'#BBB9FF', u'#ffff00')
#    colors = (u'#ff0000', u'#228b22', u'#0000cd', u'#ffa500')
#    #Colors are Red, ....write here...
#    for i in range(k):
#        xy_max = np.max(cluster_d[i], axis = 0)     #max two points, x,y in cluster i 
#        xy_min = np.min(cluster_d[i], axis = 0)    #min two points, x,y in cluster i 
#        cov_mat = np.cov(cluster_d[i][:,0],cluster_d[i][:,1])
#        meanc = np.mean(cluster_d[i], axis = 0)
#        probc = len(cluster_d[i]) / tot_rows
##        print xy_max, xy_min
#        plt.scatter(cluster_d[i][:,0], cluster_d[i][:,1],c = colors[i], s = 10)
#        dx = 0.005
#        upts = np.arange(xy_min[0], xy_max[0], (xy_max[0] - xy_min[0]) * dx) #find all points from xmin to xmax in steps of 0.005
#        vpts = np.arange(xy_min[1], xy_max[1], (xy_max[1] - xy_min[1]) * dx) #find all points from ymin to ymax in steps of 0.005
#        xpts, ypts = np.meshgrid(upts, vpts) #build a mesh using the above points
#        zpts = []
#        for i in range(len(xpts)):
#            zx = [] 
#            for j in range(len(ypts)):
#                gx = find_gix([xpts[i][j], ypts[i][j]], meanc, cov_mat, probc)
#                zx.append(gx)
#            zpts.append(zx)
#        plt.contour(xpts,ypts,zpts, 4)

#def find_gix(Xi,Mui,cov_matrix, probC):
#    x = np.subtract(Xi,Mui)
#    xt = x.T
#    inv_covmat = np.linalg.inv(cov_matrix)
#    det_cov = np.linalg.det(cov_matrix)
#    if det_cov == 0:
#        det_cov = 0.0000001
#    nix =0
#    nix = -(1 / 2) * np.matmul(np.matmul(xt, inv_covmat), x) - (1 / 2) * np.log(det_cov) - np.log(2*np.pi)    #+ np.log(probC)
#    return nix             
                    
                    
                    
                    
                    