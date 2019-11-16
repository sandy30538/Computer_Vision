import numpy as np
import cv2
import sys
import random
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def main():

    # file path
    data_path   = sys.argv[1]           # ./hw2-3_data
    output_img = sys.argv[2]           # ./output_kmeans.png

    # Making the PCA eigen vector
    num_of_training_data = 280
    num_of_pixel = 2576

    training_data = np.zeros((2576, 280))    # shape (2576,280)
    for i in range(40):
        for j in range(7):
            training_data[:,i*7+j] = cv2.imread(data_path+'/'+str(i+1)+'_'+str(j+1)+'.png', cv2.IMREAD_GRAYSCALE).astype(np.float).flatten()

    # Covariance Matrix
    mean_face = np.mean(training_data, axis=1)
    x_minus_mean = np.zeros((num_of_pixel,num_of_training_data))
    for i in range(num_of_training_data):
        x_minus_mean[:,i] = training_data[:,i] - mean_face
    covariance_matrix = np.matmul(x_minus_mean, np.transpose(x_minus_mean))/(num_of_training_data - 1)

    print("Calculating eigen vectors...")
    eigen_value, eigen_vector = np.linalg.eig(covariance_matrix)
    eigen_value = eigen_value.real
    eigen_vector = eigen_vector.real

    index_array = np.argsort(eigen_value)
    sorted_eigen_vector = np.zeros((num_of_pixel,num_of_pixel))
    for k in range(num_of_pixel):
        sorted_eigen_vector[:,k] = eigen_vector[:,index_array[num_of_pixel-1-k]]

    print("Project the Data to 10-dim...")

    # Reading the Data
    training_data = np.zeros((2576, 70))    # shape (2576,70)
    for i in range(10):
        for j in range(7):
            training_data[:,i*7+j] = cv2.imread(data_path+'/'+str(i+1)+'_'+str(j+1)+'.png', cv2.IMREAD_GRAYSCALE).astype(np.float).flatten()
    
    # Project to eigen vectors (dim = 10)
    eigen_vector_10 = np.transpose(sorted_eigen_vector[:,0:10]) #(10,2576)
    
    data_minus_mean = np.zeros((2576,70))
    for i in range(70):
        data_minus_mean[:,i] = training_data[:,i] - mean_face
    
    P_data = np.matmul(eigen_vector_10, data_minus_mean)    #(10,70)
    P_data = np.transpose(P_data)   #(70,10)

    # Cluster Mean Initialization
    cluster_mean = np.zeros((10,10))
    
    for i in range (10):
        #rand = random.randint(0,6) + (i*7)
        rand = i*7
        cluster_mean[i,:] = P_data[rand, :]

    '''
    for i in range (10):
        plt.scatter(P_data[i*7:(i*7+7),0],P_data[i*7:(i*7+7),1])

    x = cluster_mean[:,0]
    y = cluster_mean[:,1]
    plt.scatter(x,y, s=[80,80,80,80,80,80,80,80,80,80], c="blue", marker='x')
    
    plt.title("Visualization of features and centroids", weight = 'bold').set_fontsize(14)
    plt.xlabel("Fisrt eigen vector").set_fontsize(10)
    plt.ylabel("Second eigen vector").set_fontsize(10)
    plt.savefig("Centroid_Initialization"+".png")
    plt.show()
    '''

    # Weight Parameters
    w = np.array([0.6,0.4,0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1]) #(10,)

    # K-Means Iteration
    
    distance = np.zeros((70,10))    #(70,10)
    times = 0

    while(times < 15):

        times = times + 1

        # Calculate the distance between 10 means
        for i in range (10):
            difference = P_data - cluster_mean[i,:] #(70,10)
            square_weighted = np.multiply(np.square(difference),w)  #(70,10)

            distance[:,i] = np.sum(square_weighted, axis = 1)

        min_index = np.argmin(distance, axis=1)

        # updata 10 Means
        for i in range(10):
            img_index = np.where(min_index == i)

            new_mean = np.zeros((1,10))
            if len(img_index[0]) > 0:
                for l in range(len(img_index[0])):
                    new_mean = new_mean + P_data[img_index[0][l]]

                cluster_mean[i,:] = new_mean / len(img_index[0])

        # Plot the movement
        '''
        if times < 6:
            for i in range (10):
                plt.scatter(P_data[i*7:(i*7+7),0],P_data[i*7:(i*7+7),1])
            x = cluster_mean[:,0]
            y = cluster_mean[:,1]
            plt.scatter(x,y, s=[80,80,80,80,80,80,80,80,80,80], c="blue", marker='x')
            plt.title("Visualization of features and centroids", weight = 'bold').set_fontsize(14)
            plt.xlabel("Fisrt eigen vector").set_fontsize(10)
            plt.ylabel("Second eigen vector").set_fontsize(10)
            plt.savefig("Centroid_"+ str(times+1) + ".png")
            plt.show()
        '''

    '''
    # Print the cluster mean image
    for i in range (10):
        P_data_cluster = cluster_mean[i,:] #(10,)
        P_data_cluster = P_data_cluster.reshape((1,10))

        recon_image = np.matmul(P_data_cluster, eigen_vector_10) + mean_face    #(1,2576)
        cv2.imwrite('./cluster_mean_'+ str(i+1) + '.png', recon_image.reshape(56,46).astype(np.uint8))
 
    # Reconstruct all the data
    for i in range (70):
        P_data_cluster = P_data[i,:] #(10,)
        P_data_cluster = P_data_cluster.reshape((1,10))

        recon_image = np.matmul(P_data_cluster, eigen_vector_10) + mean_face    #(1,2576)
        cv2.imwrite('./data_'+ str(i+1) + '.png', recon_image.reshape(56,46).astype(np.uint8))
    ''' 


    # Compare the output and ground truth
    print(min_index)

    for i in range (10):
        plt.scatter(P_data[i*7:(i*7+7),0],P_data[i*7:(i*7+7),1])

    x = cluster_mean[:,0]
    y = cluster_mean[:,1]
    plt.scatter(x,y, s=[80,80,80,80,80,80,80,80,80,80], c="blue", marker='x')
    plt.title("Visualization of features and centroids", weight = 'bold').set_fontsize(14)
    plt.xlabel("Fisrt eigen vector").set_fontsize(10)
    plt.ylabel("Second eigen vector").set_fontsize(10)
    plt.savefig(output_img)
    #plt.show()

if __name__ == '__main__':
    main()