import numpy as np
import cv2
import argparse
import time

from joint_bilateral_filter import Joint_bilateral_filter

def main():
    parser = argparse.ArgumentParser(description='Color conversion')
    parser.add_argument('--input_path', default='./testdata/2c.png', help='path of input image')
    args = parser.parse_args()

    # Input Image
    img = cv2.imread(args.input_path)                   #BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      #RGB

    # Conventional RGB2GRAY
    conventional = img_rgb[:,:,0] * 0.299 + img_rgb[:,:,1] * 0.587 + img_rgb[:,:,2] * 0.114
    conventional = conventional.astype(np.uint8)
    cv2.imwrite('conventional_2a.png',conventional)

    # Sigma Value
    sigma_s_array = np.array([1,2,3])             #int64
    sigma_r_array = np.array([0.05,0.1,0.2])      #float64
    
    # Weight candidates
    weight = np.zeros((66,5))               #float64
    axis = np.array([0.0,0.0,1.0])          #float64       
    for i in range(66):
        if(axis[0] < 0.0001):
            weight[i][0] = 0.0
        else:
            weight[i][0] = axis[0]
        weight[i][1] = axis[1]
        if (axis[2] < 0.0001):
            weight[i][2] = 0.0
        else:
            weight[i][2] = axis[2]

        if (axis[0] < 0.0001 or axis[0] == 0.0):
            axis[2] = axis[2] - 0.1
            axis[1] = 0.0
            axis[0] = 1.0 - axis[2] 
        else:
            axis[0] = axis[0] - 0.1
            axis[1] = axis[1] + 0.1

    # Run 9 different sigma
    for i in range(3):
        for j in range(3):
            print(i," ",j)
            sigma_s = sigma_s_array[i]
            sigma_r = sigma_r_array[j]

            # Create JBF Class
            JBF = Joint_bilateral_filter(sigma_s, sigma_r, border_type='reflect')
            
            # Original Bilateral Filter
            bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)      #uint8 (0~255)
            bf_out = bf_out.astype(np.float64)

            # 66 kinds of weight combination
            for k in range(66):
                if (k % 10 == 0):
                    print("k = ", k)
                # Make guidance (float64)
                guidance = img_rgb[:,:,0] * weight[k][0] + img_rgb[:,:,1] * weight[k][1] + img_rgb[:,:,2] * weight[k][2]
                
                # Joint Bilateral Filter
                jbf_out = JBF.joint_bilateral_filter(img_rgb, guidance).astype(np.uint8)
                jbf_out = jbf_out.astype(np.float64)
                
                # Compute Error
                error = np.sum(np.abs(bf_out - jbf_out))
                weight[k][3] = error
                
            # Vote the local minima
            is_local_min = [True]*66        #list
            for k in range(11):
                for m in range(k+1):
                    index = ((k*k + k)/2) + m
                    right_index = index + 1
                    left_bottom_index = ((k+2)*(k+1)/2) + m
                    right_bottom_index = left_bottom_index + 1
                    
                    # See three points
                    if (k != m):
                        # no bottom only right
                        if(k == 10):
                            if (weight[int(index)][3] < weight[int(right_index)][3]):
                                is_local_min[int(right_index)] = False
                            else:
                                is_local_min[int(index)] = False
                        else:
                            if (weight[int(index)][3] < weight[int(right_index)][3]):
                                is_local_min[int(right_index)] = False
                            else:
                                is_local_min[int(index)] = False

                            if (weight[int(index)][3] < weight[int(left_bottom_index)][3]):
                                is_local_min[int(left_bottom_index)] = False
                            else:
                                is_local_min[int(index)] = False

                            if (weight[int(index)][3] < weight[int(right_bottom_index)][3]):
                                is_local_min[int(right_bottom_index)] = False
                            else:
                                is_local_min[int(index)] = False
                    # Only bottom two points
                    else:
                        # Still have bottom two
                        if(k != 10):
                            if (weight[int(index)][3] < weight[int(left_bottom_index)][3]):
                                is_local_min[int(left_bottom_index)] = False
                            else:
                                is_local_min[int(index)] = False

                            if (weight[int(index)][3] < weight[int(right_bottom_index)][3]):
                                is_local_min[int(right_bottom_index)] = False
                            else:
                                is_local_min[int(index)] = False

            weight[:,4] = weight[:,4] + is_local_min

            for n in range(66):
                if (is_local_min[n] == True):
                    print(n)
    print(weight[:,4])        

if __name__ == '__main__':
    main()
