import sys
import cv2 as cv
import numpy as np

#resize the input img with aspect ratio = width / height
## resize img if over max size 600 * 480
def ratio_check(input_img):
    height, width = input_img.shape[:2]
    if width > 600 or height > 480:
        if width > 600:
            height = int(height * 600 / width)
            width = 600
        if height > 480:
            width = int(width * 480 / height)
            height = 480
        input_img = cv.resize(input_img, [width, height], interpolation=cv.INTER_AREA)
    return input_img

#get all the keypoint of the image under a threshold equal to 400
def key_point(input_img):
    input_img = ratio_check(input_img) #resize input img
    XYZ = cv.cvtColor(input_img, cv.COLOR_BGR2XYZ)
    Y = XYZ[:,:,1] #get y channel of img

    sift = cv.SIFT_create() #use sift to get keypoints and descriptors
    kp, des = sift.detectAndCompute(Y, None)
    return kp, des

#basically all the process of task1, print the number of keypoints and display the image result
def task1(input):
    img = cv.imread(input)
    img = ratio_check(img)
    img2 = img.copy()
    kp, des = key_point(img)

    img2 = cv.drawKeypoints(img2,kp,None, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # add cross at the keypoint
    for k_point in kp:
        cv.drawMarker(img2, (int(k_point.pt[0]), int(k_point.pt[1])), color=(0, 0, 255), markerType=cv.MARKER_CROSS, markerSize = 10, thickness=1)
    output_img = np.concatenate((img,img2), axis=1)

    print('# of keypoints in ' + 'img01.jpg' + ' is ' + str(len(kp)))

    cv.imshow('main', output_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#count the # of keypoint
def keypoint_count(input_img, fname):
    kp, des = key_point(img)
    print('# of keypoints in ' + fname + ' is ' + str(len(kp)))
    return len(kp)

# count chi^2 distance of pair of imgs and return result
def chi_squr_distance(X, Y):
    chi_square = 0.5 * np.sum([(((a - b) ** 2) / (a + b)) if a + b else 0
                        for (a, b) in zip(X, Y)])
    return chi_square

#if input images more than 1 run task 2
if len(sys.argv)-1 > 1:
    k_count = 0
    list_of_des = []
    for i in range(1,len(sys.argv)):
        img = cv.imread(sys.argv[i])
        k_count += keypoint_count(img, sys.argv[i])
        _,des = key_point(img)   #get descriptors
        list_of_des.append(des)   #record all descriptors in the list
    all_des = np.concatenate(list_of_des) #get all descriptors from the list
    split_indexes = []
    index = 0

    for d in list_of_des:
        index += len(d)
        split_indexes.append(index)
    split_indexes = split_indexes[:-1]
    total_points = all_des.shape[0]
    print()
    print(f'total number of keypoionts={total_points}') #print # of total keypoinys
    print()

#set k value at 0.05, 0.1, 0.2
    for p in [5, 10, 20]:
        k = round(total_points * p / 100)
        print(f'K={p}%*(total number of keypoionts)={k}')
        print()
        # define criteria max_iter = 100 epsilon = 0.01
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.01)
        #apply Kmeans
        _, labels, visual_words = cv.kmeans(all_des, k, None,
                                             criteria,
                                             10, cv.KMEANS_RANDOM_CENTERS) #set flags
        labels = labels.T[0]
        split_labels = np.split(labels, split_indexes)
        unique_labels = sorted(list(set(labels)))
        normalized_histograms = [] #set normalized list of histograms
        #normalize histograms
        for img_labels in split_labels:
            histogram = np.array([sum(img_labels == ul) for ul in unique_labels])
            normalized_histogram = histogram / histogram.sum()
            normalized_histograms.append(normalized_histogram) #record normalized hist into list

        dim = len(normalized_histograms)
        distances = np.zeros((dim, dim))
        for y, y_hist in enumerate(normalized_histograms):
            for x, x_hist in enumerate(normalized_histograms):
                # input normalized histograms to calculate chi_squared distance
                distance = chi_squr_distance(y_hist, x_hist)
                distances[y, x] = distance

        print('Dissimilarity Matrix') #print title of matrix
        avg = np.sum(distances) / (dim ** 2 - dim)

        #print header of input imgs
        header = []
        header.append("")
        for rank in range(1,len(sys.argv)):
            header.append(sys.argv[rank])
        for a in range(0, len(header)):
            print(f'{header[a]:<9s}', end='   ')
        print('')

        #print chi squared distance for each pair imgs
        for y in range(dim):
            print(f'{sys.argv[y+1]:<10s}', end='')
            for x in range(dim):
                if x < y:
                    print('\t\t', end='')
                    continue
                distance = distances[y, x]
                output = f'{distance:.2f}'
                print(output, end='\t\t')
            print()
        print()

else: #import only one img  run task 1
    task1(sys.argv[1])
