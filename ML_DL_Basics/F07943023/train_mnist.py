import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ConvNet, Fully
from data import get_dataloader

if __name__ == "__main__":
    # Specifiy data folder path and model type(fully/conv)
    folder, model_type = sys.argv[1], sys.argv[2]
    
    # Get data loaders of training set and validation set
    train_loader, val_loader = get_dataloader(folder, batch_size=32)

    # Specify the type of model
    if model_type == 'conv':
        model = ConvNet()
    elif model_type == 'fully':
        model = Fully()

    # Set the type of gradient optimizer and the model it update 
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Choose loss function
    criterion = nn.CrossEntropyLoss()

    # Check if GPU is available, otherwise CPU is used
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    # Four list to plot learning curve
    train_loss = []
    train_acc = []
    validation_loss = []
    validation_acc = []

    # Run any number of epochs you want
    ep = 10
    for epoch in range(ep):
        print('Epoch:', epoch)
        ##############
        ## Training ##
        ##############
        
        # Record the information of correct prediction and loss
        correct_cnt, total_loss, total_cnt = 0, 0, 0
        
        # Load batch data from dataloader
        for batch, (x, label) in enumerate(train_loader,1):
            # Set the gradients to zero (left by previous iteration)
            optimizer.zero_grad()
            # Put input tensor to GPU if it's available
            if use_cuda:
                x, label = x.cuda(), label.cuda()
            # Forward input tensor through your model
            out = model(x)
            # Calculate loss
            loss = criterion(out, label)
            # Compute gradient of each model parameters base on calculated loss
            loss.backward()
            # Update model parameters using optimizer and gradients
            optimizer.step()

            # Calculate the training loss and accuracy of each iteration
            total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

            # Show the training information
            if batch % 500 == 0 or batch == len(train_loader):
                acc = correct_cnt / total_cnt
                ave_loss = total_loss / batch           
                print ('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
                    batch, ave_loss, acc))
        
        train_loss.append(ave_loss)
        train_acc.append(acc)

        ################
        ## Validation ##
        ################
        model.eval()
        # TODO
        val_correct_cnt, val_total_loss, val_total_cnt = 0, 0, 0

        for batch, (x, label) in enumerate(val_loader,1):

            # Set the gradients to zero (left by previous iteration)
            optimizer.zero_grad()

            # Put input tensor to GPU if it's available
            if use_cuda:
                x, label = x.cuda(), label.cuda()
            # Forward input tensor through your model
            out = model(x)
            # Calculate loss
            loss = criterion(out, label)

            # Calculate the training loss and accuracy of each iteration
            val_total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            val_total_cnt += x.size(0)
            val_correct_cnt += (pred_label == label).sum().item()

            # Show the training information
            if batch % 500 == 0 or batch == len(val_loader):
                acc = val_correct_cnt / val_total_cnt
                ave_loss = val_total_loss / batch           
                print ('Validation batch index: {}, val loss: {:.6f}, acc: {:.3f}'.format(
                    batch, ave_loss, acc))
        
        validation_loss.append(ave_loss)
        validation_acc.append(acc)
        
        model.train()

    # Save trained model
    torch.save(model.state_dict(), './checkpoint/%s.pth' % model.name())

    # Plot Learning Curve
    # TODO
    fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    axs[0, 0].plot(train_loss)
    axs[0, 0].set_xlabel('epoch', fontsize=12)
    axs[0, 0].set_ylabel('loss', fontsize=12)
    axs[0, 0].set_title('Training Loss', fontsize=14)

    axs[0, 1].plot(validation_loss)
    axs[0, 1].set_xlabel('epoch', fontsize=12)
    axs[0, 1].set_ylabel('loss', fontsize=12)
    axs[0, 1].set_title('Validation Loss', fontsize=14)

    axs[1, 0].plot(train_acc)
    axs[1, 0].set_xlabel('epoch', fontsize=12)
    axs[1, 0].set_ylabel('accuracy', fontsize=12)
    axs[1, 0].set_title('Training Accuracy', fontsize=14)

    axs[1, 1].plot(validation_acc)
    axs[1, 1].set_xlabel('epoch', fontsize=12)
    axs[1, 1].set_ylabel('accuracy', fontsize=12)
    axs[1, 1].set_title('Validation Accuracy', fontsize=14)
    
    plt.savefig('./curve_' + str(model_type) +'.png')
