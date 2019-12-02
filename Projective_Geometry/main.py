import numpy as np
import cv2
import time

# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    
    #A = np.zeros((2*N, 8))
	# if you take solution 2:
    A = np.zeros((2*N, 9))
    b = np.zeros((2*N, 1))
    H = np.zeros((3, 3))
    # TODO: compute H from A and b
    # consturct A matrix and there are N points
    for i in range(N):
        A[2*i:(2*i+2),:] = np.array([[u[i,0],u[i,1],1,0,0,0,-u[i,0]*v[i,0],-u[i,1]*v[i,0],-v[i,0]],[0,0,0,u[i,0],u[i,1],1,-u[i,0]*v[i,1],-u[i,1]*v[i,1],-v[i,1]]])

    # Using SVD
    U, S, Vt = np.linalg.svd(A)

    # Construct H matrix
    v = np.transpose(Vt)
    h = v[:,-1]
    H = h.reshape((3,3))
    return H


# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    h, w, ch = img.shape
    # TODO: some magic

    # Trun the canvas corners into (y,x)
    corners[:, [1, 0]] = corners[:, [0, 1]]
    
    # Set the image corners
    img_corners = np.array([[0,0],[0,w-1],[h-1,0],[h-1,w-1]])

    # Solve the homography
    H = solve_homography(img_corners,corners)

    # Change the picture         
    for img_x in range(w):
        for img_y in range(h):
            t = np.transpose(np.array([[img_y,img_x,1]]))
            result = np.matmul(H,t)
            result_y = np.int(np.round(result[0,0]/result[2,0]))
            result_x = np.int(np.round(result[1,0]/result[2,0]))

            canvas[result_y,result_x,:] = img[img_y,img_x,:]

# Bilinear Interpolation
def bilinear(img,x,y):
    Q12 = img[int(np.floor(y)),int(np.floor(x)),:]
    Q22 = img[int(np.floor(y)),int(np.ceil(x)),:]
    Q11 = img[int(np.ceil(y)),int(np.floor(x)),:]
    Q21 = img[int(np.ceil(y)),int(np.ceil(x)),:]
    
    x1 = int(np.floor(x))
    x2 = int(np.ceil(x))
    y1 = int(np.ceil(y))
    y2 = int(np.floor(y))

    if ((x2-x1)*(y2-y1) == 0):
        pixel = Q11*(x2-x)*(y2-y) + Q21*(x-x1)*(y2-y) + Q12*(x2-x)*(y-y1) + Q22*(x-x1)*(y-y1)
    else:
        k = 1 / ((x2-x1)*(y2-y1))
        pixel = k * (Q11*(x2-x)*(y2-y) + Q21*(x-x1)*(y2-y) + Q12*(x2-x)*(y-y1) + Q22*(x-x1)*(y-y1))
    return pixel

def main():
    # Part 1
    ts = time.time()
    canvas = cv2.imread('./input/Akihabara.jpg')
    img1 = cv2.imread('./input/lu.jpeg')    #(1073,1280,3)
    img2 = cv2.imread('./input/kuo.jpg')    #(945,1280,3)
    img3 = cv2.imread('./input/haung.jpg')  #(1281,720,3)
    img4 = cv2.imread('./input/tsai.jpg')   #(675,1200,3)
    img5 = cv2.imread('./input/han.jpg')    #(1440,2160,3)

    # shape: (4,2)
    canvas_corners1 = np.array([[779,312],[1014,176],[739,747],[978,639]])
    canvas_corners2 = np.array([[1194,496],[1537,458],[1168,961],[1523,932]])
    canvas_corners3 = np.array([[2693,250],[2886,390],[2754,1344],[2955,1403]])
    canvas_corners4 = np.array([[3563,475],[3882,803],[3614,921],[3921,1158]])
    canvas_corners5 = np.array([[2006,887],[2622,900],[2008,1349],[2640,1357]])
    
    # TODO: some magic
    # Transform the image to canvas
    transform(img1,canvas,canvas_corners1)
    transform(img2,canvas,canvas_corners2)
    transform(img3,canvas,canvas_corners3)
    transform(img4,canvas,canvas_corners4)
    transform(img5,canvas,canvas_corners5)

    cv2.imwrite('part1.png', canvas)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

    # Part 2
    ts = time.time()
    img = cv2.imread('./input/QR_code.jpg')
    # TODO: some magic
    # img = (3024, 4032, 3)
    h, w, ch = img.shape
            
    QR_coners = np.array([[1237,1978],[1209,2041],[1396,2023],[1364,2085]])
    
    output_size = 200
    output_corner = np.array([[0,0],[0,output_size-1],[output_size-1,0],[output_size-1,output_size-1]])
    output2 = np.zeros((output_size,output_size,3))
    
    # Solve homography
    H = solve_homography(QR_coners,output_corner)
    H_inv=np.linalg.inv(H)

    # Output the pictures
    for output_x in range(output_size):
        for output_y in range(output_size):
            t = np.transpose(np.array([[output_y,output_x,1]]))
            result = np.matmul(H_inv,t)
            result_y =(result[0,0]/result[2,0])
            result_x =(result[1,0]/result[2,0])

            if result_x < w-1 and result_y < h-1 and result_x >= 0 and result_y >= 0:
                output2[output_y,output_x,:] = bilinear(img,result_x,result_y)

    cv2.imwrite('part2.png', output2)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))
    
    # Part 3
    ts = time.time()
    img_front = cv2.imread('./input/crosswalk_front.jpg')   #(407, 725, 3)
    # Ground Truth (393, 549, 3)
    # TODO: some magic  
    h_input, w_input, ch_input = img_front.shape
    
    output3=np.zeros(np.shape(img_front))
    h_3, w_3, ch_3 = output3.shape

    # Define the corners
    img_front_corner = np.array([[158,344],[158,380],[229,340],[229,386]])
    output_corner = np.array([[88,338],[88,376],[243,338],[243,376]])    

    # Solving the homography
    H_3 = solve_homography(img_front_corner, output_corner)
    H_inv_3 = np.linalg.inv(H_3)
    
    # Output the pictures
    for output_x in range(w_3):
        for output_y in range(h_3):
            t = np.transpose(np.array([[output_y,output_x,1]]))
            result = np.matmul(H_inv_3,t)
            result_y =(result[0,0]/result[2,0])
            result_x =(result[1,0]/result[2,0])

            if result_x < w_3-1 and result_y < h_3-1 and result_x >= 0 and result_y >= 0:
                output3[output_y,output_x,:] = bilinear(img_front,result_x,result_y)

    cv2.imwrite('part3.png', output3)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

if __name__ == '__main__':
    main()
