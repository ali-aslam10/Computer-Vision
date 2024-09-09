#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''

There are four general steps for deep learning models:

1.    Prepare the data

2.    Build the model

3.    Train the model

4.    Analyze the model's results





Today Lecture

1. PyTorch import

2. Creat fully connected Network

3. Set device

4. Hyperparameters

5. load data

6. Initalize NW

7. Load and optimization

8. Train NW

9. Check accuracy on training and test to see how good our model



Resources:

1. https://aladdinpersson.medium.com/pytorch-neural-network-tutorial-7e871d6be7c4



use the following code and do it for NN instead of ConvNet

2. https://cs230.stanford.edu/blog/handsigns/



3. #Deeplizard:

#https://www.youtube.com/watch?v=v5cngxo4mIg&list=PLZbbT5o_s2xrfNyHZsM6ufI0iZENK9xgG



'''

#First we need will need a couple of different packages



import torch  #The top-level PyTorch package and tensor library.

import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions

import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.

import torch.nn.functional as F # All functions that don't have any parameters, like relu etc

from torch.utils.data import DataLoader # Gives easier dataset managment and creates mini batches



'''

#torch.utils.data:          Extract: Access data from source, and create minibatches

torch.util.data.DataLoader 1.Extract data from source(and load provide its access) and creat minibatches

torch.utils.data.Datasaet  2.If we make our owndataset then import ,and implement its abstract functions=>__getitem__() and __len__().

'''



import torchvision

import torchvision.datasets as datasets # Has standard datasets we can import in a nice and easy way

import torchvision.transforms as transforms # Perform transformations on dataset (convert numpy to tensor data)





batch_size = 64

train_dataset = datasets.MNIST(

    root="dataset/",

    train=True,

    transform=transforms.ToTensor(),

    download=True,

)

#traing dataset: 60,000 examples

train_loader = DataLoader( # 1)extract the dataset from source and 3) load it in form of batches.

    dataset=train_dataset, batch_size=batch_size, shuffle=True)



# tensor shape: [64 1 28 28] [batchSize noChannels height width]



#testing dataset: 10,000 examples

test_dataset = datasets.MNIST(

    root="dataset/",

    train=False,

    transform=transforms.ToTensor(),

    download=True,

)

test_loader = DataLoader(

    dataset=test_dataset, batch_size=batch_size, shuffle=True

)





class NN(nn.Module):  #1. nn.Module is the base class of all neural network models,

                      # we need to extend nn.Module and defined our subclass like NN

                      # our model should be the subclass of nn.Module

                      #2. define layers of subclass

                      #3. implement forward()



    def __init__(self, input_size, num_classes): # constructor of NN with its attributes

        super(NN, self).__init__() # calling constructor of base class

                                    # create two layer NN, first layer with 50 neural and second/output layers with 10 neuraons

        self.fc1 = nn.Linear(input_size, 50) #self.fc1.weight.shape =  50,input_size

        self.fc2 = nn.Linear(50, num_classes)

        # callable objects

    def forward(self, x):  # we must provid imp of forward () of nn.Module in our subclass

        x = F.relu(self.fc1(x)) # //can do F.softmax(self.fc1(x))

        #x = self.fc2(x)  #         x = F.softmax(self.fc2(x), dim=1)

        x = F.softmax(self.fc2(x))#, dim=1)

        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



input_size = 784 # 1x28x28 = 784 size of MNIST images (grayscale)

num_classes = 10

learning_rate = 0.001

num_epochs = 2



# create NN object and move it to device



'''

When we initialize the model the weights and biases

of the model will be initialized under the hood of PyTorch

and if you want a customized weight initialization it can be added in the NN class.

'''



model = NN(input_size=input_size, num_classes=num_classes).to(device)



'''

The standard loss function for classifications tasks in PyTorch is the CrossEntropyLoss()

which applies the softmax functionand negative log likelihood given the predictions

of the model and data labels.

'''

criterion = nn.CrossEntropyLoss()



optimizer = optim.Adam(model.parameters(), lr=learning_rate)





for epoch in range(num_epochs):

    print(f"Epoch: {epoch}")

    for batch_idx, (data, targets) in enumerate(train_loader):

        # The enumerate() method adds a counter to an iterable

        #and returns it (the enumerate object)





        # Get data to cuda if possible

        data = data.to(device=device)

        targets = targets.to(device=device)



        # Get to correct shape, 1x28x28->784

        # -1 will flatten all outer dimensions into one

        #print (data.shape) # [64, 1, 28, 28]

        #print (targets.shape) # 64 scalar values.

        data = data.reshape(data.shape[0], -1) #[64,1x28x28]=[64, 784]

        #print (data.shape) #[64,784]



        # forward propagation

        scores = model(data) #automatically call the forward method,

                                #as model is a callable object

        loss = criterion(scores, targets) # compute cost/loss on 64 example



        # zero previous gradients

        optimizer.zero_grad()



        # back-propagation

        loss.backward()



        # gradient descent or adam step

        optimizer.step()





# In[ ]:





# In[3]:


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval() # 1. our model deactivates all the layers (eg.batch normalization/dropout)
    with torch.no_grad(): #2.  not make computational graph
        for x, y in loader:
            #print (x.shape)
            x = x.to(device=device)
            y = y.to(device=device)

            x = x.reshape(x.shape[0], -1)
            print(x.shape)
            #print (y.shape)

            scores = model(x)
            print(scores.shape)

            _, predictions = scores.max(1) #. it return max value and its index, 1 mean see column-wise

            num_correct += (predictions == y).sum() # compare prediction with y, if equal sum them to count the number of same values
            num_samples += predictions.size(0)  #64, get no of samples
            break  # just to see the results for a single patch
        print(
            f"Got {num_correct} / {num_samples} with accuracy"
            f" {float(num_correct) / float(num_samples) * 100:.2f}"
        )
print ("Test accuracy: ")
check_accuracy(test_loader, model)


# In[8]:


model.train()# set mode for training,
print ("Train accuracy: ")
check_accuracy(train_loader, model)
print(train_dataset[0])


# In[5]:


print ("Test accuracy: ")
check_accuracy(test_loader, model)


# In[ ]:




