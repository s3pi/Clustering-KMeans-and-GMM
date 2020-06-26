import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.ndimage import imread
 
def main():
	#extract_Scene_Images()
	extract_Cell_Images_train()
	#unit_test_Cell_Images_train_set()
	extract_Cell_Images_test()

def unit_test_Cell_Images_train_set():
	all_images_training_set_path = "/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/Cell_Images/Train"
	count = 0
	for each_img_file_name in listdir(all_images_training_set_path):
		img_path = join(all_images_training_set_path, each_img_file_name)
		img = imread(img_path)
		count += (img.shape[0] - 6) * (img.shape[1] - 6)

	lines_count = sum(1 for line in open("/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/Cell_Images/Train_Feature_Vectors/2d_feature_vectors.txt"))
	print(count == lines_count)

def extract_Cell_Images_train():
	all_images_training_set_path = "/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/Cell_Images/Train"
	txt_file_path = "/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/Cell_Images/Train_Feature_Vectors/2d_feature_vectors.txt"
	for each_img_file_name in listdir(all_images_training_set_path):
		img_path = join(all_images_training_set_path, each_img_file_name)
		img = imread(img_path)
		feature_vectors = extract_2d_feature_vectors(img)
		f=open(txt_file_path, "a+")
		for each in feature_vectors:
			f.write(str(each) + "\n")
		f.close()
		print(sum(1 for line in open(txt_file_path)))
	
def extract_Cell_Images_test():
	all_images_test_set_path = "/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/Cell_Images/Test"
	all_txt_files_path = "/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/Cell_Images/Test_Feature_Vectors"
	for each_img_file_name in listdir(all_images_test_set_path):
		img_path = join(all_images_test_set_path, each_img_file_name)
		img = imread(img_path)
		file_name, ext = each_img_file_name.split(".")
		txt_file_path = join(all_txt_files_path, file_name + ".txt")
		feature_vectors = extract_2d_feature_vectors(img)
		f = open(txt_file_path, "w+")
		for each in feature_vectors:
			f.write(str(each) + "\n")
		f.close()
		#testing if all lines are included
		lines_count = sum(1 for line in open(txt_file_path))
		if lines_count != (img.shape[0] - 6) * (img.shape[1] - 6):
			print("alert")
		
def extract_2d_feature_vectors(img):
	feature_vectors = []
	rows, cols = img.shape
	for i in range(rows - 6):
		for j in range(cols - 6):
			patch = img[i:i+7, j:j+7]
			mean = round(np.mean(patch), 6)
			variance = round(np.var(patch), 6)
			feature_vectors.append([mean, variance])
	return feature_vectors
			

def extract_Scene_Images():
	Images_Folder = ["Train", "Test"]
	Feature_Vectors_Folder = ["Train_Feature_Vectors", "Test_Feature_Vectors"]
	dir_path = "/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/Scene_Images/"
	for i in range(2):
		train_folder = ["coast", "industrial_area", "pagoda"]
		for scene_type in train_folder:
			all_images_path = join(join(dir_path, Images_Folder[i]), scene_type)
			all_txt_files_path = join(dir_path, Feature_Vectors_Folder[i], scene_type)
			for each_img_file_name in listdir(all_images_path):
				img_path = join(all_images_path, each_img_file_name)	
				img = cv2.imread(img_path)
				file_name, ext = each_img_file_name.split(".")
				txt_file_path = join(all_txt_files_path, file_name + ".txt")
				extract_24d_feature_vectors(img, txt_file_path)

def extract_24d_feature_vectors(img, txt_file_path):
	f=open(txt_file_path, "w+")
	rows, cols, colors = img.shape
	for i in range(rows//32):
		for j in range(cols//32):#create 32 * 32 patches
			feature_vector = [0] * 24
			for patch_row in range(i*32, i*32+32):
				for patch_col in range(j*32, j*32+32):#scan through each 32 * 32 patch
					bin_no_red = img[patch_row][patch_col][0] // 32
					bin_no_blue = 8 + img[patch_row][patch_col][1] // 32
					bin_no_green = 16 + img[patch_row][patch_col][2] // 32
					feature_vector[bin_no_red] += 1
					feature_vector[bin_no_blue] += 1
					feature_vector[bin_no_green] += 1
			#testing if all pixels are read
			if np.sum(feature_vector) != 32*32*3:
				print("alert")
			f.write(str(feature_vector) + "\n")

	if rows % 32 > 0 or cols % 32 > 0:
		handle_border_pixels(img, txt_file_path, f)

	f.close()
	#checking number of patches in a image.
	print("no.of lines in file", sum(1 for line in open(txt_file_path)))
	

def handle_border_pixels(img, txt_file_path, f):
	rows, cols, colors = img.shape
	#handling inadequate columns
	for i in range(rows//32):
		feature_vector = [0] * 24
		for patch_row in range(i*32, i*32+32):
			for patch_col in range(-1, -32, -1):
				bin_no_red = img[patch_row][patch_col][0] // 32
				bin_no_blue = 8 + img[patch_row][patch_col][1] // 32
				bin_no_green = 16 + img[patch_row][patch_col][2] // 32
				feature_vector[bin_no_red] += 1
				feature_vector[bin_no_blue] += 1
				feature_vector[bin_no_green] += 1
		f.write(str(feature_vector) + "\n")

	#handling inadequate rows
	for j in range(cols//32):
		feature_vector = [0] * 24
		for patch_col in range(j*32, j*32+32):
			for patch_row in range(-1, -32, -1):
				bin_no_red = img[patch_row][patch_col][0] // 32
				bin_no_blue = 8 + img[patch_row][patch_col][1] // 32
				bin_no_green = 16 + img[patch_row][patch_col][2] // 32
				feature_vector[bin_no_red] += 1
				feature_vector[bin_no_blue] += 1
				feature_vector[bin_no_green] += 1
		f.write(str(feature_vector) + "\n")

	#handling last corner kutty patch of the image
	for i in range(-1, -32, -1):
		for j in range(-1, -32, -1):
			feature_vector = [0] * 24
			bin_no_red = img[patch_row][patch_col][0] // 32
			bin_no_blue = 8 + img[patch_row][patch_col][1] // 32
			bin_no_green = 16 + img[patch_row][patch_col][2] // 32
			feature_vector[bin_no_red] += 1
			feature_vector[bin_no_blue] += 1
			feature_vector[bin_no_green] += 1
	f.write(str(feature_vector) + "\n")


main()