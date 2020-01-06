import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import sys
import math
from sklearn.feature_extraction import image

def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Pre-processing
    # Histogram Equalization of RGB Images
    Il = hisEqulColor(Il.astype(np.uint8))
    Ir = hisEqulColor(Ir.astype(np.uint8))

    # Normalization
    normalizedIl = np.zeros((h, w, ch))
    normalizedIr = np.zeros((h, w, ch))
    normalizedIl = cv2.normalize(Il, normalizedIl, 0, 255, cv2.NORM_MINMAX)
    normalizedIr = cv2.normalize(Ir, normalizedIr, 0, 255, cv2.NORM_MINMAX)

    # Symmetric Padding
    radius = 2
    window_size = 2*radius + 1 

    padded_Il = np.pad(Il, ((radius,radius),(radius,radius),(0,0)),'symmetric')
    padded_Ir = np.pad(Ir, ((radius,radius),(radius,radius),(0,0)),'symmetric')    

    # >>> Cost computation
    # TODO: Compute matching cost from Il and Ir
    print(">>> Cost computation")
    # Create cost volume 
    max_cost = window_size*window_size - 1
    cost_volume = np.full((h, w, max_disp+1), max_cost)

    # Calculat census cost
    for row in range(h):
        for col in range(w):
            window_left = padded_Il[row:(row+window_size), col:(col+window_size), :]
            for disparity in range(max_disp+1):
                if (col-disparity) >= 0:
                    window_right = padded_Ir[row:(row+window_size), (col-disparity):(col-disparity+window_size), :]
                    
                    binary_left = window_left <= window_left[radius,radius,:]
                    binary_right = window_right <= window_right[radius,radius,:]
                    # Calculate hamming distance 
                    distance = np.sum(np.bitwise_xor(binary_left, binary_right))
                    cost_volume[row,col,disparity] = distance
  
    # >>> Cost aggregation
    # TODO: Refine cost by aggregate nearby costs
    # Guided filter
    normalizedcost = np.zeros((h, w, max_disp+1))
    normalizedcost = cv2.normalize(cost_volume, normalizedcost, 0, 255, cv2.NORM_MINMAX)
    guided_cost = cv2.ximgproc.guidedFilter(guide=normalizedIl.astype(np.uint8), src=normalizedcost.astype(np.uint8), radius=11, eps=150, dDepth=-1)

    # >>> Disparity optimization
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    # Winner-take-all
    disparity_map = np.argmin(guided_cost, axis=2)

    # >>> Disparity refinement
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    # Left-right consistency check
    print(">>> Left-right consistency check")
    # Compute the disparity map for right image
    cost_volume_right = np.full((h, w, max_disp+1), max_cost)

    # Calculat census cost for right image
    for row in range(h):
        for col in range(w):
            window_right = padded_Ir[row:(row+window_size), col:(col+window_size), :]
            for disparity in range(max_disp+1):
                if (col+disparity+window_size) <= (w + radius):
                    window_left = padded_Il[row:(row+window_size), (col+disparity):(col+disparity+window_size), :]
                    # Binary Pattern
                    binary_left = window_left <= window_left[radius,radius,:]
                    binary_right = window_right <= window_right[radius,radius,:]
                    # Calculate hamming distance 
                    distance = np.sum(np.bitwise_xor(binary_left, binary_right))
                    cost_volume_right[row,col,disparity] = distance

    # Guided filter
    normalizedcost_right = np.zeros((h, w, max_disp+1))
    normalizedcost_right = cv2.normalize(cost_volume_right, normalizedcost_right, 0, 255, cv2.NORM_MINMAX)
    guided_cost_right = cv2.ximgproc.guidedFilter(guide=normalizedIr.astype(np.uint8), src=normalizedcost_right.astype(np.uint8), radius=11, eps=150, dDepth=-1)

    # Winner-take-all
    disparity_map_right = np.argmin(guided_cost_right, axis=2)

    # Check if Dl(x,y) = Dr(x-Dl(x,y),y)
    check_map = np.zeros((h,w))
    for row in range(h):
        for col in range(w):
            value = disparity_map[row,col]
            right_value = disparity_map_right[row, col-value]
            if (value == right_value):
                check_map[row,col] = 1
            else:
                check_map[row,col] = 0
    
    holed_disparity_map = disparity_map * check_map

    # Hole-filling
    print(">>> Hole-filling")
    # Padding the maximum
    padded_hole_disparity_map = np.pad(holed_disparity_map, ((0,0),(1,1)), 'constant', constant_values=max_cost)
    
    for row in range(h):
        for col in range(1, w+1):
            if (padded_hole_disparity_map[row,col] == 0.0):
                count_from_left = 1
                count_from_right = 1
                # From Left
                while(padded_hole_disparity_map[row, (col-count_from_left)] == 0.0):
                    if ((col-(count_from_left+1)) < 0):
                        break
                    else:
                        count_from_left = count_from_left + 1
                value_left = padded_hole_disparity_map[row, (col-count_from_left)]

                # From Right
                while(padded_hole_disparity_map[row, (col+count_from_right)] == 0.0):
                    if (col+(count_from_right+1) > (w+1)):
                        break
                    else:
                        count_from_right = count_from_right + 1
                value_right = padded_hole_disparity_map[row, (col+count_from_right)]

                # Fill the hole
                if (value_left <= value_right):
                    padded_hole_disparity_map[row,col] = value_left
                else:
                    padded_hole_disparity_map[row,col] = value_right
    # Recover to the previous size
    padded_hole_disparity_map = padded_hole_disparity_map[:,1:w+1]

    # Weighted median filter
    print(">>> Weighted median filter")
    filter_radius = 2
    filter_window_size = 2*filter_radius + 1

    # Padding
    padded_result = np.pad(padded_hole_disparity_map, ((filter_radius,filter_radius),(filter_radius,filter_radius)), 'symmetric') #(290,386)

    # Filtering h = 288, w = 384
    sigma = 0.25
    for row in range(h):
        for col in range(w):
            window = padded_result [row:(row+filter_window_size), col:(col+filter_window_size)]
            
            distance_matrix = np.zeros((filter_window_size, filter_window_size))
            for y in range(filter_window_size):
                for x in range(filter_window_size):
                    distance_matrix[y,x] = math.sqrt((y-filter_radius)**2 + (x-filter_radius)**2) 
            
            weighted_window = np.exp( -1*(distance_matrix + np.absolute(window-window[filter_radius,filter_radius]))/sigma )
            weighted_window = weighted_window.flatten()
            
            # Flatten the window
            window = window.flatten()

            # Convolution
            window_result = window * weighted_window
            
            # Sort the window
            order = np.argsort(window_result)
            threshold = np.sum(window_result) / 2

            # Choose the weighted median
            tmp_sum = 0
            for k in range(window_result.shape[0]):
                if ((tmp_sum + window_result[order[k]]) >= threshold):
                    padded_result[(row+filter_radius),(col+filter_radius)] = window[order[k]]
                    break
                else:
                    tmp_sum = tmp_sum + window_result[order[k]]
    
    # Output
    labels = padded_result[filter_radius:(h+filter_radius), filter_radius:(w+filter_radius)]

    return labels.astype(np.uint8)