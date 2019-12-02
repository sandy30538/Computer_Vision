import numpy as np
import cv2
import sys

def main(ref_image,template,video):
    # Load in the input image
    ref_image = cv2.imread(ref_image)  ## load gray if you need.
    ref_image = cv2.resize(ref_image, (410, 410))   # Change to the same size of marker
    h, w, ch = ref_image.shape
    ref_point = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2) #(4,1,2)

    # Load in the marker image
    template = cv2.imread(template, cv2.IMREAD_GRAYSCALE)  ## load gray if you need.

    # Load in the reference video
    video = cv2.VideoCapture(video)
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) #1080,1920
    film_fps = video.get(cv2.CAP_PROP_FPS)  #30.01172

    # Write out the output video
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videowriter = cv2.VideoWriter("ar_video.mp4", fourcc, film_fps, (film_w, film_h))

    # Create the sift
    sift = cv2.xfeatures2d.SIFT_create()

    # Find the keypoints and descriptors 
    keypoint, descriptor = sift.detectAndCompute(template,None)
    
    i = 0
    while(video.isOpened()):
        ret, frame = video.read()
        print('Processing frame {}'.format(i))
        if ret:  ## check whethere the frame is legal, i.e., there still exists a frame
            ## TODO: homography transform, feature detection, ransanc, etc.

            # Change the frame into gray scale
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)     #(1080,1920)

            # Find the keypoints and descriptor in gray frame
            # gray descriptor (1331,128)
            gray_keypoint, gray_descriptor = sift.detectAndCompute(gray_frame,None)
            
            # BFMatcher 
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptor, gray_descriptor, k=2)

            # Apply Ratio test
            good = []
            for m,n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            #(225,1,2)
            marker_points = np.float32([(keypoint[m.queryIdx].pt) for m in good]).reshape(-1,1,2)
            frame_points = np.float32([(gray_keypoint[m.trainIdx].pt) for m in good]).reshape(-1,1,2)

            # Find Homography (source, destination)
            H, status = cv2.findHomography(marker_points, frame_points, cv2.RANSAC, 5.0)

            # Do the transform
            target_point = cv2.perspectiveTransform(ref_point,H)    #(4,1,2)
            ref_point_4_2 = np.reshape(ref_point,(4,2))
            tar_point_4_2 = np.reshape(target_point,(4,2))

            # Calculate the perspective transform from four pairs
            M = cv2.getPerspectiveTransform(ref_point_4_2, tar_point_4_2)
            
            # Apply a perspective transformation to an image
            output = cv2.warpPerspective(ref_image,M,(film_w, film_h))  #(1080,1920,3)

            # Remove the marker part 
            sum_output = np.sum(output,axis=2)
            sum_not_0 = np.where(sum_output!=0)            

            not_0_y = np.array(sum_not_0[0])
            not_0_x = np.array(sum_not_0[1])

            for j in range(not_0_y.shape[0]):
                frame[not_0_y[j], not_0_x[j], :] = np.array([0,0,0]).T
            
            # Paste on the reference image
            result = frame + output
            
            # Write the frame to the video
            videowriter.write(result)

            i = i + 1
        else:
            break     
            
    video.release()
    videowriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ## you should not change this part
    ref_path = './input/sychien.jpg'
    template_path = './input/marker.png'
    video_path = sys.argv[1]  ## path to ar_marker.mp4
    main(ref_path,template_path,video_path)