import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def main():

    # file path
    data_path   = sys.argv[1]           # ./hw2-3_data
    output_path = sys.argv[2]           # ./pca_output/

    # Problem 3(a)-1
    # Read in the training data
    print("Problem 3(a)-1")
    num_of_training_data = 280
    num_of_pixel = 2576

    training_data = np.zeros((num_of_pixel,num_of_training_data))    # shape (2576,280)
    for i in range(40):
        for j in range(7):
            training_data[:,i*7+j] = cv2.imread(data_path+'/'+str(i+1)+'_'+str(j+1)+'.png', cv2.IMREAD_GRAYSCALE).astype(np.float).flatten()
    
    # Calculate the Mean Face
    mean_face_output = np.mean(training_data, axis=1).reshape((56,46))
    mean_face_output = mean_face_output.astype(np.uint8)
    cv2.imwrite(output_path + '/mean_face.png', mean_face_output)

    # Covariance Matrix
    mean_face = np.mean(training_data, axis=1)

    x_minus_mean = np.zeros((num_of_pixel,num_of_training_data))
    for i in range(num_of_training_data):
        x_minus_mean[:,i] = training_data[:,i] - mean_face

    covariance_matrix = np.matmul(x_minus_mean, np.transpose(x_minus_mean))/(num_of_training_data - 1)

    print("Calculating eigen vector...")
    eigen_value, eigen_vector = np.linalg.eig(covariance_matrix)
    eigen_value = eigen_value.real
    eigen_vector = eigen_vector.real

    index_array = np.argsort(eigen_value)
    sorted_eigen_vector = np.zeros((num_of_pixel,num_of_pixel))
    for k in range(num_of_pixel):
        sorted_eigen_vector[:,k] = eigen_vector[:,index_array[num_of_pixel-1-k]]

    # Output the first five eigenfaces
    for i in range(5):
        normalized_minus = sorted_eigen_vector[:,i]-(sorted_eigen_vector[:,i].min())
        normalized_divide = normalized_minus / normalized_minus.max()                   #(2576,)
        eigen_face = (normalized_divide*255).astype(np.uint8).reshape(56,46)

        cv2.imwrite(output_path + '/eigenface_'+ str(i+1) + '.png', eigen_face)

    print("===============================")

    # Problem 3(a)-2
    print("Problem 3(a)-2")
    # Read in Image 8-6 (2576,)
    image_8_6 = cv2.imread(data_path+'/8_6.png', cv2.IMREAD_GRAYSCALE).astype(np.float).flatten()

    # Eigen vector Sets
    eigen_vector_280 = np.transpose(sorted_eigen_vector[:,0:280])   #(280,2576)

    # Project to Eigen vector
    P = np.matmul(eigen_vector_280,(image_8_6-mean_face))   #(280,)
    P = P.reshape((1,280))

    n = np.array([5,50,150,280])
    print("Output the reconstructed image 8_6")
    for i in range(4):
        reconstruct_image_8_6 = np.matmul(P[:,0:n[i]],eigen_vector_280[0:n[i],:]) + mean_face    #(1,2576)

        cv2.imwrite(output_path + '/reconstructed_8_6_'+ str(n[i]) + '.png', reconstruct_image_8_6.reshape(56,46).astype(np.uint8))
        
        mse = np.sum((image_8_6.reshape((1,2576)) - reconstruct_image_8_6)**2).astype(np.float)/2576
        print("n = " + str(n[i]) + "  mse = " + str(mse))
   
    print("===============================")
    # Problem 3(a)-3
    print("Problem 3(a)-3")

    # Reading the testing Data
    testing_data = np.zeros((num_of_pixel,120))    # shape (2576,120)
    for i in range(40):
        for j in range(7,10):
            testing_data[:,i*3+(j-7)] = cv2.imread(data_path+'/'+str(i+1)+'_'+str(j+1)+'.png', cv2.IMREAD_GRAYSCALE).astype(np.float).flatten()

    # Project to eigen vectors (dim = 100)
    eigen_vector_100 = np.transpose(sorted_eigen_vector[:,0:100])   #(100,2576)

    testing_data2 = np.zeros((num_of_pixel,120))
    for k in range(120):
        testing_data2[:,k] = testing_data[:,k] - mean_face
    
    P_testing = np.matmul(eigen_vector_100,testing_data2)   #(100,120)

    tsne_output = TSNE(n_components=2).fit_transform(np.transpose(P_testing))   #(120,2)

    label = np.repeat(np.arange(0,40),3)
    plt.scatter(tsne_output[:,0], tsne_output[:,1], cmap = 'tab20', c=label)
    plt.title("Distribution of test images", weight = 'bold').set_fontsize(14)
    plt.xlabel("Dimension 1").set_fontsize(10)
    plt.ylabel("Dimension 2").set_fontsize(10)
    #plt.show()
    plt.savefig(output_path + '/tsne_testing.png')

    print("===============================")
    # Problem 3(a)-4
    print("Problem 3(a)-4")

    # Gram Matrix
    # (280,280)
    covariance_matrix_G = np.matmul(np.transpose(x_minus_mean),x_minus_mean)

    print("Calculating eigen vector for Gram Matrix...")
    eigen_value_G, eigen_vector_G = np.linalg.eig(covariance_matrix_G)
    eigen_value_G = eigen_value_G.real
    eigen_vector_G = eigen_vector_G.real    #(280,280)

    index_array_G = np.argsort(eigen_value_G)  #(280,)
    sorted_eigen_vector_G = np.zeros((num_of_training_data,num_of_training_data))   #(280,280)
    for k in range(num_of_training_data):
        sorted_eigen_vector_G[:,k] = eigen_vector_G[:,index_array_G[num_of_training_data-1-k]]

    # Recover the eigen vector
    recovered_eigen_vector_G = np.matmul(x_minus_mean, sorted_eigen_vector_G)   #(2576,280)
    recovered_eigen_vector_G = recovered_eigen_vector_G / np.linalg.norm(recovered_eigen_vector_G, axis = 0)    #(2576,280)
    recovered_eigen_vector_G = np.transpose(recovered_eigen_vector_G) #(280,2576)

    # Project to Eigen vector
    P_G = np.matmul(recovered_eigen_vector_G,(image_8_6-mean_face))   #(280,)
    P_G = P_G.reshape((1,280))

    print("Output the reconstructed image 8_6 from Gram Matrix...")
    for i in range(4):
        reconstruct_image_8_6_G = np.matmul(P_G[:,0:n[i]],recovered_eigen_vector_G[0:n[i],:]) + mean_face    #(1,2576)

        cv2.imwrite(output_path + '/G_reconstructed_8_6_'+ str(n[i]) + '.png', reconstruct_image_8_6_G.reshape(56,46).astype(np.uint8))
        
        mse = np.sum((image_8_6.reshape((1,2576)) - reconstruct_image_8_6_G)**2).astype(np.float)/2576
        print("n = " + str(n[i]) + "  mse = " + str(mse))

    print("===============================")
    # Problem 3(b) k-NN
    print("Problem 3(b) k-NN")

    # Labeling
    # training Data (2576,280)
    # testing Data (2576,120)
    training_label = np.zeros(280).astype(np.float) #(280,)
    testing_label = np.zeros(120).astype(np.float)  #(120,)
    for i in range(40):
        for j in range(7):
            training_label[i*7+j] = i
            if (j < 3):
                testing_label[i*3+j] = i

    training_data_minus_mean = np.zeros((num_of_pixel,280))    #(2576,280)
    testing_data_minus_mean = np.zeros((num_of_pixel,120))    #(2576,120)
    for m in range(280):
        training_data_minus_mean[:,m] = training_data[:,m] - mean_face
        if (m < 120):
            testing_data_minus_mean[:,m] = testing_data[:,m] - mean_face        

    # Parameters
    dim = np.array([3,10,39])
    k = np.array([1,3,5])
    accuracy = np.zeros(9)
    
    # Best Dim and k
    max_score = 0.0
    best_dim = 0.0
    best_k = 0.0

    # Training Data Project on to eigen vector and use K-NN to bulid classifier
    for i in range(3):
        for j in range(3):
            eigen_vector_knn = np.transpose(sorted_eigen_vector[:,0:dim[i]])   #(Dim,2576)
            
            P_knn = np.matmul(eigen_vector_knn,training_data_minus_mean)   #(Dim,280)
            P_knn = np.transpose(P_knn) #(280, Dim)

            # Train a k-NN classifier
            knn_classifier = KNeighborsClassifier(n_neighbors = k[j])
            
            # 3-fold cross-validation
            three_fold = cross_val_score(knn_classifier, P_knn, training_label, cv=3, scoring='accuracy')
            accuracy[i*3+j] = three_fold.mean()

            # Find the best k and n
            if (accuracy[i*3+j] >= max_score):
                max_score = accuracy[i*3+j]
                best_dim = dim[i]
                best_k = k[j]

            print("n = " + str(dim[i]) + " k = " + str(k[j]))
            print(three_fold)
            print("accuracy = " + str(accuracy[i*3+j]))

    print("Best n = " + str(best_dim) + ", Best k = " + str(best_k))

    # Test on the testing data
    eigen_vector_knn_test = np.transpose(sorted_eigen_vector[:,0:best_dim])   #(39,2576)
    
    P_knn = np.matmul(eigen_vector_knn_test,training_data_minus_mean)   #(39,280)
    P_knn = np.transpose(P_knn) #(280, 39)

    P_knn_test = np.matmul(eigen_vector_knn_test,testing_data_minus_mean)   #(39,120)
    P_knn_test = np.transpose(P_knn_test) #(120, 39)

    # Build the best classifier
    knn_classifier_test = KNeighborsClassifier(n_neighbors = best_k)
    knn_classifier_test.fit(P_knn, training_label)

    # Recognition Rate
    print('Recognition rate: ' + str(knn_classifier_test.score(P_knn_test, testing_label)))


if __name__ == '__main__':
    main()