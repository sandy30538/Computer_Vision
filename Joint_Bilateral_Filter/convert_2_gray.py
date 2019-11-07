import numpy as np
import cv2
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='Color conversion')
    parser.add_argument('--input_path_a', default='./testdata/2a.png', help='path of input image')
    parser.add_argument('--input_path_b', default='./testdata/2b.png', help='path of input image')
    parser.add_argument('--input_path_c', default='./testdata/2c.png', help='path of input image')
    args = parser.parse_args()

    # Input Image
    img_a = cv2.imread(args.input_path_a)                   #BGR
    img_a_rgb = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)      #RGB
    cv2.imwrite("img_a_rgb.png", img_a_rgb)
    img_b = cv2.imread(args.input_path_b)
    img_b_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB) 
    cv2.imwrite("img_b_rgb.png", img_b_rgb)
    img_c = cv2.imread(args.input_path_c)
    img_c_rgb = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB) 
    cv2.imwrite("img_c_rgb.png", img_c_rgb)

    # Conventional RGB2GRAY
    conventional_a = img_a_rgb[:,:,0] * 0.299 + img_a_rgb[:,:,1] * 0.587 + img_a_rgb[:,:,2] * 0.114
    conventional_a = conventional_a.astype(np.uint8)
    cv2.imwrite('conventional_2a.png',conventional_a)
    conventional_b = img_b_rgb[:,:,0] * 0.299 + img_b_rgb[:,:,1] * 0.587 + img_b_rgb[:,:,2] * 0.114
    conventional_b = conventional_b.astype(np.uint8)
    cv2.imwrite('conventional_2b.png',conventional_b)
    conventional_c = img_c_rgb[:,:,0] * 0.299 + img_c_rgb[:,:,1] * 0.587 + img_c_rgb[:,:,2] * 0.114
    conventional_c = conventional_c.astype(np.uint8)
    cv2.imwrite('conventional_2c.png',conventional_c)

    # Advanced RGB2GRAY
    a_y1 = (img_a_rgb[:,:,0] * 0.0 + img_a_rgb[:,:,1] * 1.0 + img_a_rgb[:,:,2] * 0.0).astype(np.uint8)
    a_y2 = (img_a_rgb[:,:,0] * 0.0 + img_a_rgb[:,:,1] * 0.0 + img_a_rgb[:,:,2] * 1.0).astype(np.uint8)
    a_y3 = (img_a_rgb[:,:,0] * 1.0 + img_a_rgb[:,:,1] * 0.0 + img_a_rgb[:,:,2] * 0.0).astype(np.uint8)

    b_y1 = (img_b_rgb[:,:,0] * 1.0 + img_b_rgb[:,:,1] * 0.0 + img_b_rgb[:,:,2] * 0.0).astype(np.uint8)
    b_y2 = (img_b_rgb[:,:,0] * 0.8 + img_b_rgb[:,:,1] * 0.0 + img_b_rgb[:,:,2] * 0.2).astype(np.uint8)
    b_y3 = (img_b_rgb[:,:,0] * 0.7 + img_b_rgb[:,:,1] * 0.0 + img_b_rgb[:,:,2] * 0.3).astype(np.uint8)

    c_y1 = (img_c_rgb[:,:,0] * 1.0 + img_c_rgb[:,:,1] * 0.0 + img_c_rgb[:,:,2] * 0.0).astype(np.uint8)
    c_y2 = (img_c_rgb[:,:,0] * 0.9 + img_c_rgb[:,:,1] * 0.1 + img_c_rgb[:,:,2] * 0.0).astype(np.uint8)
    c_y3 = (img_c_rgb[:,:,0] * 0.0 + img_c_rgb[:,:,1] * 0.0 + img_c_rgb[:,:,2] * 1.0).astype(np.uint8)
    
    # Write out the images
    cv2.imwrite('2a_y1.png',a_y1)
    cv2.imwrite('2a_y2.png',a_y2)
    cv2.imwrite('2a_y3.png',a_y3)

    cv2.imwrite('2b_y1.png',b_y1)
    cv2.imwrite('2b_y2.png',b_y2)
    cv2.imwrite('2b_y3.png',b_y3)

    cv2.imwrite('2c_y1.png',c_y1)
    cv2.imwrite('2c_y2.png',c_y2)
    cv2.imwrite('2c_y3.png',c_y3)

if __name__ == '__main__':
    main()
