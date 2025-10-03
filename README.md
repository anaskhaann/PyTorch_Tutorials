These are my notes while Learning PyTorch.

Uploaded Here just to take a guick look through.

---

## **What are Tensors**

Tensor is a Specialized Multi dimesntional Array designed for mathematical and computational efficiency.

1. Scalars : 0 - dimensional Tensors (a single number)

Single value often used for simple metrics or constants.

Example: Loss Value

2. Vectors: 1-dimensional tensors (a list of numbers)

Represents a sequence or a collection of values.

- Example: Feature vector: In natural language processing, each word in a sentence may be represented as a 1D vector using embeddings.

Example: [0.12, -0.84, 0.33] (a word embedding vector from a pre-trained model like Word2Vec or Glove).

3. Matrices: 2-dimensional tensors (a 2D grid of numbers)
   Represents tabular or grid-like data.

Example: Grayscale images: A grayscale image can be represented as a 2D tensor, where each entry corresponds to the pixel intensity.
Example:
[[0, 255, 128],
[34, 90, 180]]

---

4. 3D Tensors: **Coloured images**
   Adds a third dimension, often used for stacking data.

Example: RGB Images: A single RGB image is represented as a 3D tensor (width × height × channels).

Examples: RGB Image (e.g., 256x256): Shape [256, 256, 3]

---

5. 4D Tensors: Batches of RGB images
   Adds the batch size as an additional dimension to 3D data.

Example: Batches of RGB Images: A dataset of coloured images is represented as a 4D tensor (batch size × width × height × channels).

Example: A batch of 32 images, each of size 128x128 with 3 colour channels (RGB),
would have shape [32, 128, 128, 3].
(32 images, width of images, height of images, number of channels i.e RGB-3 )

6. 5D Tensors: Video data
   Adds a time dimension for data that changes over time (e.g., video frames).

Video Clips: Represented as a sequence of frames, where each frame is an RGB
image.

Example: A batch of **10** video clips, each with **16** frames of size **64x64** and **3** channels (RGB), would have shape [10, 16, 64, 64, 3]

---

## **Why Are Tensors Useful?**

1. **Mathematical Operations**:

- Tensors enable efficient mathematical computations (addition, multiplication, dot product, etc.) necessary for neural network operations.

2. **Representation of Real-world Data**:

- Data like images, audio, videos, and text can be represented as tensors:
- **Images**: Represented as 3D tensors (width × height × channels).
- **Text**: Tokenized and represented as 2D or 3D tensors (sequence length × embedding
  size).

3. **Efficient Computations**:

- Tensors are optimized for hardware acceleration, allowing computations on GPUs or TPUs, which are crucial for training deep learning models.

---

## **Where are the Tensors Used in Deep Learning**

1. **Data Storage**:

- Training data (images, text, etc.) is stored in tensors.

2. **Weights and Biases**:

- The learnable parameters of a neural network (weights, biases) are stored as tensors.

3. **Matrix Operations**:

- Neural networks involve operations like matrix multiplication, dot products, and
  broadcasting—all performed using tensors.

4. **Training Process**:

- During forward passes, tensors flow through the network.
- Gradients, represented as tensors, are calculated during the backward pass.

---

## **Data Types Available In PyTorch**

| **Data Type**                  | **Dtype**          | **Description**                                                                                                                      |
| ------------------------------ | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| **32-bit Floating Point**      | `torch.float32`    | Standard floating-point type used for most deep learning tasks. Provides a balance between precision and memory usage.               |
| **64-bit Floating Point**      | `torch.float64`    | Double-precision floating point. Useful for high-precision numerical tasks but uses more memory.                                     |
| **16-bit Floating Point**      | `torch.float16`    | Half-precision floating point. Commonly used in mixed-precision training to reduce memory and computational overhead on modern GPUs. |
| **BFloat16**                   | `torch.bfloat16`   | Brain floating-point format with reduced precision compared to `float16`. Used in mixed-precision training, especially on TPUs.      |
| **8-bit Floating Point**       | `torch.float8`     | Ultra-low-precision floating point. Used for experimental applications and extreme memory-constrained environments (less common).    |
| **8-bit Integer**              | `torch.int8`       | 8-bit signed integer. Used for quantized models to save memory and computation in inference.                                         |
| **16-bit Integer**             | `torch.int16`      | 16-bit signed integer. Useful for special numerical tasks requiring intermediate precision.                                          |
| **32-bit Integer**             | `torch.int32`      | Standard signed integer type. Commonly used for indexing and general-purpose numerical tasks.                                        |
| **64-bit Integer**             | `torch.int64`      | Long integer type. Often used for large indexing arrays or for tasks involving large numbers.                                        |
| **8-bit Unsigned Integer**     | `torch.uint8`      | 8-bit unsigned integer. Commonly used for image data (e.g., pixel values between 0 and 255).                                         |
| **Boolean**                    | `torch.bool`       | Boolean type, stores `True` or `False` values. Often used for masks in logical operations.                                           |
| **Complex 64**                 | `torch.complex64`  | Complex number type with 32-bit real and 32-bit imaginary parts. Used for scientific and signal processing tasks.                    |
| **Complex 128**                | `torch.complex128` | Complex number type with 64-bit real and 64-bit imaginary parts. Offers higher precision but uses more memory.                       |
| **Quantized Integer**          | `torch.qint8`      | Quantized signed 8-bit integer. Used in quantized models for efficient inference.                                                    |
| **Quantized Unsigned Integer** | `torch.quint8`     | Quantized unsigned 8-bit integer. Often used for quantized tensors in image-related tasks.                                           |

---

## **Inplace Operation**

Any result of the operation is saved as new tensor in memory. But this can be very in efficient if we have a large tensor to store in memory.

So to solve this we can store this result to existing variable. Just Like `Inplace` parameter in Pandas.

To do that we just need to add `_` infront of any function.

Example:

- `relu` is normal (torch.relu(m))
- `relu_` is inplace operation (m.relu\_())

---

## **Reshaping Tensors**

- transpose
- reshape (product of reshape should be equal to original dimension)
- flatten
- permute (Apply Permutation, that is we can simply change the shape as per our needed index, We can pass that we need current 2nd on 3rd, current 1st on 2nd and current 3rd on 1st)
- unsqueeze (add a new dimension (one) to given postion) : This is usefull at images passing in batch. Use to manipulate dimensions.
- squeeze (remove a dimension (one) from given position, default all). Use to manipulate dimensions.

---

## **Why AutoGrad**

Because calculating derative of chains is very difficult.

Because for a simple chain rule of derivation we have to perform various mathematical operations.

So in neural networks while doing backward propogation to calculate the **loss with respect to Parameters**

Because loss is depent on the predicted value, Predicted value is depend on the weight and bias, weight and bias is depend on the input. So in order to find loss w.r.t to weight and loss w.r.t biass is quite Tough

**The Basic Process of Training the Neural Network Includes**

1. **Forward pass** - Compute the output of the network given an input.

2. **Calculate loss** - Calculate the loss function to quantify the error.

3. **Backward pass** - Compute gradients of the loss with respect to the
   parameters.

4. **Update gradients** - Adjust the parameters using an optimization
   algorithm (e.g., gradient descent).

> Thus AutoGrad comes in Picture. It is a feature in Pytorch which helps to Calculate the Derivates Automatically.

---

# **AutoGrad**

Autograd is a core component of PyTorch that provides automatic differentiation for tensor operations.

It enables gradient computation, which is essential for training machine learning
models using optimization algorithms like gradient descent.

## **For 1st**

**Now calculate the Derivative**

For this square function our derivate will be `2x`

Just run the command backward on Y to calculate `dy/dx`.

If we needed to find `dx/dy` we would run backward on x given that the relationship is satisfying.

---

## **Clearing Grad**

Why we need to Clear.

Because If we run it again and again all gradients will get accumulated rather than replacing.

Previous Gradient gets added to current Gradient.

---

## **How to Disable gradient Tracking**

When it is usefull.

- After training when only doing Predictions. Because at training we need backward pass while prediction we dont need backward pass.

- requires_grad => False
- detach()
- torch.no_grad()

**with `requires_grad=False`**

---

## **How we Build Training Pipeline in PyTorch.**

1. Load the dataset
2. Basic preprocessing (Scaling, Encoding)
3. Training Process

- Create the model
- Forward pass
- Loss calculation
- Backprop
- Parameters update (using Gradient Descent)

4. Model evaluation using (Accuracy)

---

## **NN Module**

The torch.nn module in PyTorch is a core library that provides a wide array of classes and functions designed to help developers build neural networks efficiently and effectively.

It abstracts the complexity of creating and training neural networks by offering pre-built layers, loss functions, activation functions, and other utilities, enabling you to focus on designing and experimenting with model architectures.

---

## **Key Components of torch.nn:**

1. **Modules (Layers):**

- `nn.Module`: The base class for all neural network modules. Your custom models and layers should subclass this class.

- `Common Layers`: Includes layers like nn.Linear (fully connected layer), nn.Conv2d (convolutional layer), nn.LSTM (recurrent layer), and many others.

2. **Activation Functions:**

- Functions like nn.ReLU, nn.Sigmoid, and nn.Tanh introduce non-linearities to the
  model, allowing it to learn complex patterns.

3. **Loss Functions:**

- Provides loss functions such as nn.CrossEntropyLoss, nn.MSELoss, and nn.NLLLoss to quantify the difference between the model's predictions and the actual targets.

4. Container Modules:

- nn.Sequential: A sequential container to stack layers in order.

5. **Regularization and Dropout:**

- Layers like nn.Dropout and nn.BatchNorm2d help prevent overfitting and improve
  the model's ability to generalize to new data.

---

## **Torch Optim Module**

torch.optim is a module in PyTorch that provides a variety of optimization
algorithms used to update the parameters of your model during training.

It includes common optimizers like Stochastic Gradient Descent (SGD), Adam,
RMSprop, and more.

It handles weight updates efficiently, including additional features like learning rate scheduling and weight decay (regularization).

The model.parameters() method in PyTorch retrieves an iterator over all the
trainable parameters (weights and biases) in a model.
These parameters are instances of torch.nn.Parameter and include:

- Weights: The weight matrices of layers like nn.Linear, nn.Conv2d, etc.
- Biases: The bias terms of layers (if they exist).

The optimizer uses these parameters to compute gradients and update them
during training.

---

## **Model Building Using nn.Module**

**Now changes will be Made Here**

- use nn.Module
- change constructor and forward pass function to use nn.module attributes.
- Use Built in Loss Function
- Dont calculate and update gradient manually use built in optimizations.
- Using Stocastic Gradient Descent. This requires all parameters and learning rate.

The `model.parameters()` method in PyTorch retrieves an iterator over all the trainable parameters (weights and biases) in a model. These parameters are instances of torch.nn.Parameter and include:

- **Weights**: The weight matrices of layers like nn.Linear, nn.Conv2d, etc.
- **Biases**: The bias terms of layers (if they exist).

The optimizer uses these parameters to compute gradients and update them during training.

## **Why we need Dataset Class and DataLoader**

---

The code which we have written previously have a **Big Flaw**

```py
for epoch in range(epochs):

  # Forward Pass
  y_pred = model.forward(x_train_tensor)

  # Loss Calculate
  loss = loss_function(y_pred.squeeze(), y_train_tensor)

  # Make gradeints zero(Doing it before backward pass bcoz it is suggested.)
  optimizer.zero_grad()

  # BackWard Loss(backPropogate)
  loss.backward()

  # Update weight & bias using optimizer
  optimizer.step()

  # Print the Loss
  print(f"Epoch:{epoch +1}, Loss:{loss}")
  print('='*60)

```

In the above code we are looping again & again over the entire dataset i.e(x_train_tensor).

- This is very inefficient way. Because we are using Batch Gradient Descent.
- Convergence is not that great. Because we are updating the parameter one time by looking at the overall data.

---

**Solution** : Divide data into Batches and perform on that. And iterate over the batch. This is called as **Mini Batch Gradient Descent.**

---

**Solution one by using nested loops to iterate over data**

This will work but there are some problems with this approach.

- Sometime data gathering is very difficult, because suppose there are dataset for images in multiple folder based on categories. So this Approach does not handle that.

- Another Problem is There is no transformation in this. Sometime for **RGB** images we required to transform some. Suppose convert colour images to **B/W**.

- No Shuffling and Sampling

- Batch management & Parallelization.

So this will work but it is not good way.

**To Solve this problems we have DataSet & DataLoader Classess**

---

## **How Dataset & DataLoader Works**

Dataset and DataLoader are core abstractions in PyTorch that decouple how you define your data from how you efficiently iterate over it in training loops.

**Dataset Class**

The Dataset class is essentially a blueprint. When you create a custom Dataset, you decide how data is loaded and returned.

It defines:

- `__init__()` which tells how data should be loaded.

- `__len__()` which returns the total number of samples.This will be used to calculate number of batches based on batch size and lenght of data.

- `__getitem__(index)` which returns the data (and label) at the
  given index.

---

Dataset class does the loading(reading) of the data and it remembers where data is present in your memory.

**Dataset Class is an Abstract Class.**

Thus we need to create all the three methods in our custom datasetclass.

**DataLoader Class**

The DataLoader wraps a Dataset and handles batching, shuffling, and parallel loading for you.

DataLoader Control Flow:

- At the start of each epoch, the DataLoader (if shuffle=True)
  shuffles indices(using a sampler).

- It divides the indices into chunks of batch_size.

- for each index in the chunk of batch_size, data samples are fetched from the Dataset object using get item because it gives item based on index.

- The samples are then collected and combined into a batch (using `collate_fn`).

- The batch is returned to the main training loop.

---

DataLoader Class works on creating and extarcting batches from that loaded data which helps to create mini batch.

#### Workflow

**1.Sampler and Batch Creation (Main Process):**

Before training starts for the epoch, the DataLoader’s sampler generates a shuffled list of all 10,000 indices. These
are then grouped into 312 batches of 32 indices each. All these batches are queued up, ready to be fetched by
workers.

**2.Parallel Data Loading (Workers):**

At the start of the training epoch, you run a training loop like:

```py
for batch_data, batch_labels in dataloader:
  # Training logic
```

Under the hood, as soon as you start iterating over dataloader, it dispatches the first four batches of indices
to the four workers:

- Worker #1 loads batch 1 (indices [batch_1_indices])
- Worker #2 loads batch 2 (indices [batch_2_indices])
- Worker #3 loads batch 3 (indices [batch_3_indices])
- Worker #4 loads batch 4 (indices [batch_4_indices])

_Each worker:_

- Fetches the corresponding samples by calling **getitem** on the dataset for each index in that batch.

- Applies any defined transforms and passes the samples through collate_fn to form a single batch tensor.

**3.First Batch Returned to Main Process:**

Whichever worker finishes first sends its fully prepared batch (e.g., batch 1) back to the main process.

As soon as the main process gets this first prepared batch, it yields it to your training loop, so your code for
batch_data, batch_labels in dataloader:receives (batch_data, batch_labels) for the first batch.

**4.Model Training on the Main Process:**

While you are now performing the forward pass, computing loss, and doing backpropagation on the first
batch, the other three workers are still preparing their batches in parallel.

By the time you finish updating your model parameters for the first batch, the DataLoader likely has the
second, third, or even more batches ready to go (depending on processing speed and hardware).

**5.Continuous Processing:**

As soon as a worker finishes its batch, it grabs the next batch of indices from the queue.

For example: after Worker #1 finishes with batch 1, it immediately starts on batch 5. After Worker #2
finishes batch 2, it takes batch 6, and so forth.

This creates a pipeline effect: at any given moment, up to 4 batches are being prepared concurrently.

**6. Loop Progression:**

Your training loop simply sees:

```python
for batch_data, batch_labels in dataloader:
    # forward pass
    # loss computation
    # backward pass
    # optimizer step
```

Each iteration, it gets a new, ready-to-use batch without long I/O waits, because the workers have been pre-
loading and processing data in parallel.

**7. End of the Epoch:**

After ~312 iterations, all batches have been processed. All indices have been consumed, so the DataLoader
has no more batches to yield.

The epoch ends. If shuffle=True, on the next epoch, the sampler reshuffles indices, and the whole process
repeats with workers again loading data in parallel.

---

## **How does shuffling Happen in Data Loader**

This uses sampler

In PyTorch, the sampler in the DataLoader determines the strategy for selecting samples from the dataset during data loading. It controls how indices of the dataset are drawn for each batch.

**Types of Samplers**

PyTorch provides several predefined samplers, and you can create custom ones:

1. **SequentialSampler**:

- Samples elements sequentially, in the order they appear in the dataset.

- Default when shuffle=False.

- Should be used when working with timeseries data

2. **RandomSampler**:

- Samples elements randomly without replacement.

- Default when shuffle=True.

We can also create custom Sampling Strategies.

Suppose we need sampling but in that sample we also need it to follow a particular types of Distribution.

---

## **Collate Function**

The collate_fn in PyTorch's DataLoader is a function that specifies how to combine a list of samples from a dataset into a single batch.

By default, the DataLoader uses a simple batch collation mechanism, but collate_fn allows you to customize how the data should be processed and batched.

**Why do we need custom merging strategy for merging data to create batches**

- When there will be difference size of tensor in our rows and while merging they both cannot be merged because the shape is not same

---

## **Data Loader Important Parameters**

The DataLoader class in PyTorch comes with several parameters that allow you to customize how data is loaded, batched, and preprocessed. Some of the most commonly used and important parameters include:

1. dataset(mandatory) :
   The Dataset from which the DataLoader will pull data.
   Must be a subclass of torch.utils.data.Dataset that implements **getitem** and
   **len**.

2. batch_size: How many samples per batch to load. Default is 1. Larger batch sizes can speed up training on GPUs but require more memory.

3. shuffle: If True, the DataLoader will shuffle the dataset indices each epoch.
   Helpful to avoid the model becoming too dependent on the order of samples.

4. num_workers:
   The number of worker processes used to load data in parallel.
   Setting num_workers > 0 can speed up data loading by leveraging multiple CPU
   cores, especially if I/O or preprocessing is a bottleneck.

5. pin_memory:
   If True, the DataLoader will copy tensors into pinned (page-locked) memory before
   returning them.
   This can improve GPU transfer speed and thus overall training throughput,
   particularly on CUDA systems.

6. drop_last:
   If True, the DataLoader will drop the last incomplete batch if the total number of samples is not divisible by the batch size.
   Useful when exact batch sizes are required (for example, in some batch
   normalization scenarios).

7. collate_fn:
   A callable that processes a list of samples into a batch (the default simply stacks tensors).
   Custom collate_fn can handle variable-length sequences, perform custom batching logic, or handle complex data structures.

8. sampler:
   sampler defines the strategy for drawing samples (e.g., for handling imbalanced
   classes, or custom sampling strategies).
   batch_sampler works at the batch level, controlling how batches are formed.
   Typically, you don’t need to specify these if you are using batch_size and shuffle.
   However, they provide lower-level control if you have advanced requirements.

---

## **Steps to Build on GPU**

1. Check the availability of GPU
2. Move the Model to GPU
3. Modify the Training Loop by Moving Data to GPU
4. Modify the Evaluation Loop by Moving Data to GPU
5. Optimize the GPU Usage:

- Creating Bigger Batches like 64, 128 etc.

- User `pin_memory=True` to make data loading from cpu to gpu faster.

generally data is in pager memory and go to pin memory and then comes for loading.
So here we are directly keeping it on pinned memory so the time from getting page to pin and then to us is reduced.
These are just the concept of Operating System

---

## **WHY we need to Optimize**

Go to the last section of Evaluation and you will see that when performing evaluation on both the training and testing data.

The accuracy on training data is very high.

The accuracy on testing data is low as compared to training data.

**Thus this is the sign of OverFitting.**

---

## **How can we Solve this**

1.  **Adding More Data:** This we cannot do because we are already using complete data. ❌

2.  **Reducing the Complexity of NN Architecture.** We cannot do that because we are already using simple architecture. ❌

3.  **Regularization:** In this we add a Penalty term in loss function. So model tries to minimize both which reduce the overfitting. L2 regularization is used mostly in Deep Learning. ✅

4.  **Dropouts:** Randomly turn of some neurons during forward pass. While training some neurons are turned off randomly. ✅

5.  **Data Augmentation:** Modify the data such as tranform, flip, invert, etcc. But this works good with CNNs so we are not going to use this. ❌

6.  **Batch Normalization:** Normalize the batch of the data to make mean=0 and std=1. ✅

7.  **Early Stopping:** When we see that after certain epochs the loss is not reducing or else increasing then we just stop before completing all epochs.

---

## **DropOut**

1. Applied to the hidden layers.

2. Applied after the ReLU **activation function.**

3. Randomly turns off **p% neurons** in the hidden layer during each forward pass.

4. This has a regularization effect.

5. During evaluation dropout is not used.

---

## **Batch Normalization**

This improves the Training Instability. The problem of **internal covariate shifts**.

Because output of one layer acts as input to other thus they are dependent on the previous input.

Because the weights are getting changes again and again there distribution is also changed this makes training instable.

Thus we simply normalize the mini batches of data set to be in a standard range.

- **Applied to Hidden Layers:** Typically applied to the hidden layers of a neural network, but not to the output layer.

- **Applied After Linear Layers and Before Activation Functions:** Normalizes the output of the preceding layer (e.g., after nn.Linear) and is usually followed by an activation function (e.g., ReLU).

- **Normalizes Activations:** Computes the mean and variance of the activations within a mini-batch and uses these statistics to normalize the activations.

- **Includes Learnable Parameters:**
  Introduces two learnable parameters, gamma (scaling) and beta (shifting),
  which allow the network to adjust the normalized outputs.

- **Improves Training Stability:**
  Reduces internal covariate shift, stabilizing the training process and allowing
  the use of higher learning rates.

- **Regularization Effect:**
  Introduces some regularization because the statistics are computed over a
  mini-batch, adding noise to the training process.

- **Consistent During Evaluation:**
  During evaluation, BatchNorm uses the running mean and variance
  accumulated during training, rather than recomputing them from the minibatch.

---

## **L2 Regularization**

- **Applied to Model Weights:** Regularization is applied to the weights of the model **(not on bias)** to penalize large values and
  encourage smaller, more generalizable weights.

- **Introduced via Loss Function or Optimizer:** Adds a penalty term **λ∑wi2** to the loss function in L2 regularization.
  In weight decay, directly modifies the gradient update rule to include λwi, effectively shrinking weights during training.

- **Penalizes Large Weights:**
  Encourages the network to distribute learning across multiple parameters, avoiding reliance on a few large weights.

- **Reduces Overfitting:**
  Helps the model generalize better to unseen data by discouraging overly complex
  representations.

- **Controlled by a Hyperparameter:**
  A regularization coefficient (λ, often set via weight_decay in optimizers) controls the strength of the penalty. Larger values lead to stronger regularization.

- **No Effect on Bias Terms:**
  Regularization is typically applied only to weights, not biases, as biases don't directly affect model complexity.

- **Active During Training:**
  Regularization affects weight updates only during training. It does not explicitly influence the model during inference.

The most easiest way to apply regularization is to implement it through weight decays.

**In optimization step during gradient descent directly add the loss to the weight.**

---

## **WHY we need Hyper Parameter Tuning**

What ever we have chosen till now is just based on our intuition there we no logic behind it.

Example:

- Why we chose 128 neurons in hidden layers?? No reason
- Why we chose 2 hidden Layers?? No reason
- Epochs count
- Learning Rate
- Optimizer
- Batch Size and much more things.

There was no logic for choosing such values. So we need some concept/logic to decide the parameters & Model Architecture.

---

## **How can we Solve this**

**By Experimentation with HyperParameter Tuning**

There are bunch of techniques such as

1.  **Grid Search CV**

2.  **Random CV**

3.  **Bayesian Search**

We are going to use Bayesian Search because it is most advance and we can use it in both ML, DL.

The library which we are going to use to Implement this is **Optuna**

---

## **Optuna Overview**

WE have to define a `objective_function` which is a python function.

```python
def objective_function(trail_obj):
  # define seeach space
  # initalize model
  # initalize params
  # perform training loop
  # perform evaluation
  # return trail_accuracy
```

This will recieve a trail object.

1. Define the search space to find the values. Such as lowest and highest values of hidden layer. We define this for each paramters i.e for hidden layers, epochs, batches, neural per layer etc.

2. Intialize the model inside this function.
3. Intialize all parameter inside this function.
4. Perform Training Loop inside this function.
5. Perform Evaluation Loop inside this function.

This will return the accuracy for that Trail.

---

Then we create Study object which is similar to experiment. Here we define how many trails we needed.

With help of study object we create trail and send it to the objective Function.

---

## **What is CNN**

Convolutional Neural Networks (CNNs) are a type of deep learning model specifically designed to process and analyse structured data, such as images or
videos. They are particularly effective in tasks like image classification, image recognition and object detection due to their ability to automatically and
efficiently learn spatial hierarchies of features.

Cnn works well with grid like Data.

**Quick Overview of How CNN Works and Perform well on grid like data.**

The complete Architecture is divided into 2 components.

- Feature Extaction
- Classification.

#### **Feature Extraction**

In this we try to extract all the key features of the data. For example for a cat image extract all features such as _Pointy ears, Tail, Moustache, Size._

The feature Extraction consist of two layers.

1. Convolution Layer.
2. Pooling Layer.

- **Convolution Layer:** In this we have small small filters (e.g: 3x3 filter). The main task of this filter is to apply this filter on images. And they extract low level features from our data. We can have `n` number of filters (e.g 32 filters, 64 filters) Each filter will work on extracting different features.

- **Pooling Layer:** It does multiple things, But the main task is to reduce the size of the tensor by applying techniques like, Max pooling, Average Pooling etc.

#### **Classification**

Now all the extracted features of convolution and pooling are taken as input in a single fully connected layer.(1D tensor for Input).

Here we use a ANN which we are using till now.

**So In order to create feature Extraction Layer we have to specify Make it in pairs.**

- `channels` : This is we are getting from input image, so now rather than taking num_features as constructor input we are going to take channels as input. we will get this from image features.
- `padding_scheme`: it is nothing but to define which padding we are using

**Creating 1st Pair**

```python

# Create convolution layer of 1st pair
nn.Conv2d(input_channels,number_of_filters,kernel_size, padding_scheme)
# BatchNorm....
# Create Activation....
# Create Pooling layer of 1st pair
nn.MaxPool2d(kernel_size,stride)
```

**Creating 2nd Pair**

- For 2nd Pair Our input will be what we are recieving from previous layer.

**To create Classification Layer**

- apply the flattening layer to make it 1 dimensional using Flatten.
- Apply Hidden layers:

> Now to pass input to this layer we have to understand the shape of flatten layer. To get that we have to understand what we have passed in feature extraction.

> In 1st pair size of image is (1,28,28) --> convolution layer (32 filters) & since padding so size does not change only channel changes due to filters --> (32,28,28) --> pooling operation(2x2) & stride 2 thus size will get half.===>> (32,14,14)

> In 2nd pair size of image is (32,14,14) --> convolution layer (64 filters) & since padding so size does not change only channel changes due to filters --> (64,14,14) --> pooling operation(2x2) & stride 2 thus size will get half.===>> (64,7,7)

> The final tensor will have a shape (64,7,7). Thus after flattening it we will get `64x7x7` as input numbers for classification layer

We can also calculate this through code this is just explaination behind the seen.

---

## **What is Transfer Learning**

Transfer learning is a machine learning technique where a model trained on one task is reused (partially or fully) for a different but related task. Instead of training a model from scratch, which can be computationally expensive and require large datasets, transfer learning
leverages knowledge from a pre-trained model to improve learning efficiency and performance.

**Why we need it**

- Deep Learning Models are Data Hungry: And sometimes we dont have that much data for our Use Case.

- Training Deep Learning Models are Costly: Because it is GPU intensive i.e Computational Inefficient.

**How Transfer Learning Works**

1. **Pretraining on a Large Dataset:**
   A model is first trained on a large dataset (e.g., ImageNet for images, GPT for text).
   The model learns general features, such as edges and shapes in images or syntax and semantics in text.

2. **Fine-Tuning for a New Task:**
   The pre-trained model is then adapted to a new, often smaller, dataset.
   Some layers may be frozen (not updated), while others are fine-tuned for the specific
   task.

- We train the model on larger data set.
- Now train the model on small data set for `fine_tuning` but in this we only train the fully connected Layer.
- We freeze the Feature Extraction Layers because what we have learned on larger data is not loss.

**Why does it works**

Because in CNN

- starting convolution layers of CNN learns to detect edges
- and the complexity of learning those patterns increase over these convolution layers.

- Any image in the world can be identify with almost those features. Thus this works.

---

## **What are RNNs**

A Recurrent Neural Network (RNN) is a type of neural network designed for processing **sequential data**.

Unlike traditional feedforward networks, which process inputs independently,RNNs maintain a **memory** of previous inputs by using **loops in their architecture**.

This makes them well-suited for tasks where **context and order matter**, such as time series forecasting, speech recognition and text generation.

- Rnn Process the data i.e this sentence word by word.

- at time `t=1` they will process 1st word and give output.

- at t2 process word 2 but include the output of t1 and soo on.

---

The main Challenge is we dont know english we have to convert those wors to numbers.

Multiple Methods:

1. One Hot Encoding.
2. Embeddings : In this we convert words as **vector**

To maintiain Consistency we also Send one input to first time stamp.

These will be set of random numbers.

So each time stamp we are processing one word along with previous output.

This whole process is called **Unfolding Threshold**

---

## **Steps**

Load Data set

---

Convert data to Numbers. That means convert them to embeddings.

We cannot directly convert them to embeddings first we have to perform following steps.

- Form a Vocabolary. (All unique words in dataset). Store vocab in mappings with along its index.

- Provide index to sentence with referencing the index of Vocab.

- Send this sentence indexes to embedding models. Which will see those index and convert them to vectors.

---

Build the Architecture of RNN

---

Train Model on Dataset

---

Prediction

---

## **LSTM**

**NEXT WORD PREDICTION IS BASED ON LANGUAGE MODELLING**

**Language Modeling** is the task of predicting the next word (or character) in a sequence based on the context of previous words.

In Language Modelling the data is mostly unsupervised.

- The first thing is we need to convert this unsupervised data into supervised data.

- For Each word (or combination of word) we need to make input output pair.

- Convert this data into numerical features.

- Create Vocabulary.
- Replace the input output mapping with this vocabs indexes.
- Index to embeddings.
- And the same task we did previously
- The architecture will be different.

**What Happen in LSTM Cell**

- Suppose we have a 1 single input(1,34,100).

- We send a single embedding(input) at single time. Along with this we recieve cell state and hidden state as input.

- After some processing inside we get the updated cell state and updated hidden state output at each time stamp.

- If we are at the last time stamp then the last step hidden state will become the final output to user.

---

1. We send (1,34,100) at t=1, Here h=1 & c=1 will be some random number.

Here we have 34 words of each 100 dimension so we will send 1 word of 100 dimension to it.

2. Now we will recieve h=2 and c=2 after processing which will act as input to second cell and next word from our 34 words.

3. Now we will Loop this 34 times for a single sequence because our sentence have 34 words.

---

Now the same process will happen for a batch i.e In parellel for 32 set.

- 1st word of all 32 batch will go to input at t=1, after processing 2nd word of all 32 batch will go to input at t=2. and soo on.

- We will do this until all words are completed

**This was also the reason we made our inputs of same size for processing all sequence in parallel.**

---
