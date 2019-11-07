import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r, border_type='reflect'):
        
        self.border_type = border_type
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s

    def joint_bilateral_filter(self, input, guidance):
        ## TODO
    
        # Initial information
        length = input.shape[0]         #316
        width = input.shape[1]          #316
        radius = self.sigma_s*3         #9
        sqr_sigma_s = self.sigma_s**2   #9
        sqr_sigma_r = self.sigma_r**2   #0.01
        window_size = 2*radius + 1      #19

        # Symmetric padding Input and Guidance Image
        padded_input = np.pad(input,((radius,radius),(radius,radius),(0,0)),'symmetric')            # (length+2r,width+2r,3),uint8
        if (guidance.ndim == 2):
            padded_guidance = np.pad(guidance,((radius,radius),(radius,radius)),'symmetric')        
            padded_guidance = padded_guidance / 255                                                 #(length+2r,width+2r),float64
        else:
            padded_guidance = np.pad(guidance,((radius,radius),(radius,radius),(0,0)),'symmetric')  
            padded_guidance = padded_guidance / 255                                                 #(length+2r,width+2r,3),float64

        # Spatial Kernel (2-dim)
        spatial_2d = np.zeros((window_size,window_size))
        for i in range(window_size):
            for j in range(window_size):
                spatial_2d[i][j] = ((i-radius)**2 + (j-radius)**2) * (-1) / (2*sqr_sigma_s)
        spatial_2d = np.exp(spatial_2d)

        # Process the whole image
        output = np.zeros((length,width,3))

        for i in range(length):
            for j in range(width):

                # f(x-i,y-j)
                sliced_input = padded_input[i:i+window_size,j:j+window_size,:]                  #(19,19,3),float64
                
                # Range Kernel
                range_2d = np.zeros((window_size,window_size))
                if (guidance.ndim == 2):
                    sliced_guidance = padded_guidance[i:i+window_size,j:j+window_size]          #(19,19),float64                 
                    range_2d = np.exp(((sliced_guidance-sliced_guidance[radius][radius])**2) * (-1) / (2*sqr_sigma_r))
                else:
                    sliced_guidance = padded_guidance[i:i+window_size,j:j+window_size,:]        #(19,19,3),float64
                    range_2d = np.exp(((sliced_guidance[:,:,0]-sliced_guidance[radius][radius][0])**2 + (sliced_guidance[:,:,1]-sliced_guidance[radius][radius][1])**2 + (sliced_guidance[:,:,2]-sliced_guidance[radius][radius][2])**2)*(-1)/(2*sqr_sigma_r))      
                
                # Final Kernel
                kernel_2d = spatial_2d * range_2d                                               #(19,19),float64
                kernel_sum = np.sum(kernel_2d)
                kernel_3d = np.repeat(kernel_2d[..., np.newaxis], 3, axis=-1)                   #(19,19,3),float64

                # Fiiler
                filtered_input = sliced_input * kernel_3d                                       #(19,19,3),float64
                filtered_sum_r = np.sum(filtered_input[:,:,0])
                filtered_sum_g = np.sum(filtered_input[:,:,1])
                filtered_sum_b = np.sum(filtered_input[:,:,2])

                # Final Output
                output[i][j][0] = filtered_sum_r/kernel_sum
                output[i][j][1] = filtered_sum_g/kernel_sum
                output[i][j][2] = filtered_sum_b/kernel_sum

        return output
