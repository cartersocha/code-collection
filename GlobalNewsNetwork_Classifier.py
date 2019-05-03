import numpy as np 
import matplotlib.pyplot as plt
from scipy import ndimage 
from keras.datasets import mnist 
from tqdm import tqdm_notebook #used for progress bar - how far through training we are

def process_input(X,Y):
    X = np.reshape(X,(X.shape[0], 1,28,28))
    X= X/255 #normalise input features 
    Y = np.eye(10)[Y.reshape(-1)].T #Y.reshape(-1) flattens the input to a 1D array
    #in general np.eye(num_classes)[array].T will take a 1D array with m training examples, and 
    #one-hot encode it into a (num_classes,m) matrix
    
    idx = np.random.permutation(np.arange(X.shape[0])) #shuffle indices
    
    X = X[idx]
    Y = Y[:,idx]
    return X,Y

(x_train_dev, y_train_dev), (x_test, y_test) = mnist.load_data()

x_train_dev, y_train_dev = process_input(x_train_dev, y_train_dev)

x_train, y_train = x_train_dev[:-10000], y_train_dev[:,:-10000]
x_dev, y_dev = x_train_dev[-10000:], y_train_dev[:,-10000:]

x_test , y_test = process_input(x_test , y_test)

def relu(x, deriv=False):
    if deriv:
        return (x>0)
    return np.multiply(x, x>0)

def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev,W) 
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + float(b)
    return Z

def zero_to_hero(X,pad):
    
    X_padded = np.pad(X,((0,0),(0,0),(pad,pad),(pad,pad)),'constant',constant_values = 0)
    
    return X_padded

def convolution_junction(image,kernel,bias,hparameters):
    
    (image_count,number_of_channels,n_H_prev,n_W_prev) = image.shape
    (n_C_prev,n_c,f,f1) = kernel.shape

    stride = hparameters['stride']
    pad = hparameters['pad']
    
    X_pad = zero_to_hero(image,pad)
    
    output_height = int(np.floor(((n_H_prev + (2*pad) -f) /stride) + 1))
    output_width = int(np.floor(((n_W_prev+ (2*pad)-f)/stride) + 1))

    result = np.zeros(shape=(image_count,n_c,output_height,output_width))

    for t in range(0,image_count):
        
        a_prev_image = X_pad[t]
        
        for y in range(0,output_height):
            
                for z in range(0,output_width):
                    
                    for m in range(0,n_c):
                        
                        vert_start = y*stride
                        vert_end = vert_start+f
                        horiz_start = z*stride
                        horiz_end = horiz_start+f
                        
                        
                        a_slice_prev = a_prev_image[:,vert_start:vert_end,horiz_start:horiz_end]
                        
                        result[t, m, y, z] = conv_single_step(a_slice_prev,kernel[:,m,:,:],bias[0][0][0][m])
                        
    cache = (image, kernel, bias, hparameters)
    
    return result, cache

def pool_forward(x,mode="max"):
    x_patches = x.reshape(x.shape[0],x.shape[1],x.shape[2]//2, 2,x.shape[3]//2, 2)
    if mode=="max":
        out = x_patches.max(axis=3).max(axis=4)
        mask  =np.isclose(x,np.repeat(np.repeat(out,2,axis=2),2,axis=3)).astype(int)
    elif mode=="average": 
        out =  x_patches.mean(axis=3).mean(axis=4)
        mask = np.ones_like(x)*0.25
    return out,mask

def conv_backward(dZ, image, kernel,hparameters):
    ### START CODE HERE ###
    # Retrieve information from "cache"
    #(image, kernel, bias, hparameters) = cache
    pad = hparameters['pad']
    
    # Retrieve dimensions from A_prev's shape
    (m,n_c_prev,n_h_prev, n_w_prev) = image.shape
    
    # Retrieve dimensions from W's shape
    (n_c_prev, n_c, f, f,) = kernel.shape
    
    # Retrieve dimensions from dZ's shape
    (m, n_c, n_h, n_w) =dZ.shape
    
    # Initialize dA_prev, dW, db with the correct shapes
    da_prev = np.zeros(shape=(m,n_c_prev,n_h_prev,n_w_prev))
    dw = np.zeros(shape=(n_c_prev,n_c,f,f))
    db = np.zeros(shape=(1,n_c,1,1))
    
    # Pad A_prev and dA_prev
    pad_aprev = zero_to_hero(da_prev,pad)
    pad_image = zero_to_hero(image,pad)
    
    # loop over the training examples
    for i in range(m):  
        # select ith training example from A_prev_pad and dA_prev_pad
        aprev_indv = pad_aprev[i]
        image_indv = pad_image[i]
        
        for h in range(n_h):
            
            for w in range(n_w):
                
                for c in range(n_c):
                    # Find the corners of the current "slice"
                    vert_start = h
                    vert_end = h + f
                    horz_start = w
                    horz_end = w + f
                    
                    # Use the corners to define the slice from a_prev_pad
                    slicer_aprev = image_indv[:,vert_start:vert_end,horz_start:horz_end]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    aprev_indv[:,vert_start:vert_end, horz_start:horz_end] += kernel[:,c,:,:] * dZ[i, c, h, w]
                    dw[:,c,:,:] += slicer_aprev * dZ[i, c, h, w]
                    db[:,c,:,:] += dZ[i, c, h, w]
                    
        # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        da_prev[i,:,:,:] = aprev_indv[:,pad:-pad,pad:-pad]
    ### END CODE HERE ###
    
    # Making sure your output shape is correct
    assert(da_prev.shape == (m, n_c_prev, n_h_prev, n_w_prev))
    
    return da_prev, dw, db

def mask_on(x):

    ### START CODE HERE ### (≈1 line)
    mask = (x==np.max(x))
    ### END CODE HERE ###
    
    return mask

def distribute_value(dz, shape):
    ### START CODE HERE ###
    # Retrieve dimensions from shape (≈1 line)
    (n_h,n_w) = shape
    
    # Compute the value to distribute on the matrix (≈1 line)
    average = dz / (n_h*n_w)
    
    #print(average)
    # Create a matrix where every entry is the "average" value (≈1 line)
    a = np.ones(shape)*average
    ### END CODE HERE ###
    
    return a

def backstroke(dA, cache, mode = "max"):
    # Retrieve information from cache (≈1 line)
    (A_prev, hparameters) = cache
    
    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    #stride = hparameters["stride"]
    f = hparameters["f"]
    
    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    (image_count,n_C_prev,n_H_prev, n_W_prev) = A_prev.shape
    (image_count,n_c,n_h,n_w) = dA.shape
    
    # Initialize dA_prev with zeros (≈1 line)
    da_prev = np.zeros(shape=(A_prev.shape))
    
    # loop over the training examples
    for i in range(image_count):
        # select training example from A_prev (≈1 line)
        a_select = A_prev[i]
        
        for h in range(n_h):
            
            for w in range(n_w):
                
                for c in range(n_c):
                # loop over the channels (depth)
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h
                    vert_end = h + f
                    horz_start = w
                    horz_end = w + f
                    
                    if mode == "max":
                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        a_slice = a_select[c,vert_start:vert_end,horz_start:horz_end]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = mask_on(a_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        da_prev[i,c,vert_start:vert_end,horz_start:horz_end] =+ np.multiply(mask,dA[i,c,h,w])
                        
                    elif mode == "average":
                        # Get the value a from dA (≈1 line)
                        a = dA[i,c,h,w]
                        # Define the shape of the filter as fxf (≈1 line)
                        shape = (1,1)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                        da_prev[i,c,vert_start:vert_end,horz_start:horz_end] += distribute_value(a,shape)
    ### END CODE ###
    
    # Making sure your output shape is correct
    assert(da_prev.shape == A_prev.shape)

    return da_prev

def loss_function(y_pred,y,parameters,lambd):
    m = y.shape[1]
    cost = (-1/m)*np.sum(y*np.log(y_pred))
    
    regularisation_term = 0
    for key in parameters:
        if "W_" in key: #all the weights
            regularisation_term += np.sum(np.square(parameters[key]))
    
    regularised_cost = cost + (lambd/(2*m))*regularisation_term
    
    return regularised_cost
def fc_forward(x,w,b):
    return relu(w.dot(x)+b)

def softmax_forward(x,w,b):
    z = w.dot(x)+b
    z -= np.mean(z,axis=0,keepdims=True) #this ensures that the value exponentiated doesn't become too large and overflow
    a = np.exp(z) 
    a = a/np.sum(a,axis=0,keepdims=True)
    return a+1e-8 #add 1e-8 to ensure no 0 values - since log 0 is undefined

def fc_backward(dA,a,x,w,b):
    m = dA.shape[1]
    dZ = dA*relu(a,deriv=True)
    dW = (1/m)*dZ.dot(x.T)
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
    dx =  np.dot(w.T,dZ)
    return dx, dW,db

def softmax_backward(y_pred, y, w, b, x):
    m = y.shape[1]
    dZ = y_pred - y
    dW = (1/m)*dZ.dot(x.T)
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)

    dx =  np.dot(w.T,dZ)

    return dx, dW,db

def init_conv_parameters(f, n_c, k):
    
    return 0.5*np.random.normal(size=(k,n_c,f,f)), 0.15*np.ones((1,1,1,n_c))
                                                                      
def init_fc_parameters(n_x,n_y):
    return 0.1*np.random.normal(size=(n_y,n_x)),0.15+np.ones((n_y,1)) #slight positive bias to prevent dead ReLU

def initialise_parameters():    
    parameters={}
    parameters["W_conv1"], parameters["b_conv1"] = init_conv_parameters(5, 4, 1)

    parameters["W_fc1"],parameters["b_fc1"] = init_fc_parameters(784,128)
    parameters["W_softmax"],parameters["b_softmax"] = init_fc_parameters(128,10)

    return parameters

def pool_backward(dx, mask):
    return mask*(np.repeat(np.repeat(dx,2,axis=2),2,axis=3))

def forward_prop(X,parameters,hparameters): 
    cache={}
    
    cache["z_conv1"], ccc = convolution_junction(X,parameters["W_conv1"], parameters["b_conv1"],hparameters)

    cache["a_conv1"] = relu(cache["z_conv1"])
    
    cache["z_pool1"], cache["mask_pool1"] = pool_forward(cache["a_conv1"])
    
    cache["a_flatten"] = np.reshape(cache["z_pool1"], (cache["z_pool1"].shape[0],-1)).T
    
    cache["a_fc1"] = fc_forward(cache["a_flatten"],parameters["W_fc1"],parameters["b_fc1"])
    
    return softmax_forward(cache["a_fc1"],parameters["W_softmax"],parameters["b_softmax"]),cache

def accuracy(y_pred,y):
    preds = np.argmax(y_pred,axis=0) #number with highest probability
    truth = np.argmax(y,axis=0) #correct label is 1 rest are 0 so this will get correct label
    return np.mean(np.equal(preds,truth).astype(int)) #check for each one if classified correctly,then take mean

def backprop(X,Y,Y_pred,parameters,cache,lambd, hparameters):
    grads = {}
    
    dA, grads["dW_softmax"],grads["db_softmax"] =softmax_backward(Y_pred, Y, parameters["W_softmax"],
                                                                  parameters["b_softmax"],cache["a_fc1"])

    dA, grads["dW_fc1"],grads["db_fc1"] = fc_backward(dA,cache["a_fc1"],cache["a_flatten"],
                                                      parameters["W_fc1"],parameters["b_fc1"])

    dA = np.reshape(dA.T,cache["z_pool1"].shape)
    grads["dz_pool1"] = dA
    dA = pool_backward(dA, cache["mask_pool1"])
    #this is where the bug is 

    dA = dA*relu(cache["z_conv1"],deriv=True)
    grads["dz_conv1"] = dA
    grads["dx"], grads["dW_conv1"],grads["db_conv1"] = conv_backward(dA,X,parameters["W_conv1"],hparameters)
    
    #regularisation term
    for key in grads:
        if "W" in key:
            grads[key]= grads[key]+ (lambd/X.shape[0])*parameters[key[1:]] 
    return grads

#%config InlineBackend.figure_format = 'retina'
#%matplotlib notebook
def train_model(X_train, Y_train, X_dev, Y_dev,num_epochs,batch_size,lambd,learning_rate,parameters = initialise_parameters() ):
    train_costs = []
    train_evals = []
    dev_evals = []

    
    momentum = {}
    beta = 0.9
    for param in parameters:
        momentum[param] = np.zeros_like(parameters[param])
        
    hparameters = {"pad" : 2,
               "stride": 1,
               "f": 5}

    
    for epoch in tqdm_notebook(range (num_epochs), total=num_epochs,desc="Number of Epochs"):
        print("Training the model, epoch: " + str(epoch+1))
        #cycle through the entire training set in batches
        for i in tqdm_notebook(range(0,X_train.shape[0]//batch_size), total =X_train.shape[0]//batch_size, desc = "Minibatch number"):
            
            
            #get the next minibatch to train on
            X_train_minibatch = X_train[i*batch_size:(i+1)*batch_size]
            Y_train_minibatch = Y_train[:,i*batch_size:(i+1)*batch_size]
            
            
            #perform one cycle of forward and backward propagation to get the partial derivatives w.r.t. the weights
            #and biases. Calculate the cost - used to monitor training
            y_pred, cache = forward_prop(X_train_minibatch,parameters,hparameters)
            minibatch_cost = loss_function(y_pred,Y_train_minibatch,parameters,lambd)
            minibatch_grads = backprop(X_train_minibatch,Y_train_minibatch,y_pred,parameters, cache,lambd,hparameters)
                            
         
            #update the parameters using gradient descent
            for param in parameters.keys():
                momentum[param] = beta *  momentum[param] + minibatch_grads["d"+param]
                parameters[param] = parameters[param] - learning_rate* momentum[param]
            
            train_costs.append(minibatch_cost)


            
            train_eval_metric = accuracy(y_pred,Y_train_minibatch)
            train_evals.append(train_eval_metric)

            
            #periodically output an update on the current cost and performance on the dev set for visualisation
            #if(i%50 == 0):
            #    #visualise the activations and gradients
            #    visualisation(X_train_minibatch,Y_train_minibatch,cache, minibatch_grads, parameters, y_pred)
            ##    print("\n \nTraining set error: "+ str(minibatch_cost))
             #   print("Training set accuracy: "+ str(train_eval_metric))
             #   y_dev_pred,_ = forward_prop(X_dev,parameters)
             #   dev_eval_metric = accuracy(y_dev_pred,Y_dev)
             #   dev_evals.append(dev_eval_metric)
             #   print("Accuracy on dev set: "+ str(dev_eval_metric))
             #   ax3.plot(dev_evals)
             #   fig.canvas.draw()
    print("Training complete!")
    #return the trained parameters 
    return parameters

parameters =train_model(x_train,y_train,x_dev,y_dev,
                        num_epochs=2,batch_size=128,lambd=2,learning_rate=1e-3)