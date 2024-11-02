# business-intelligence-project
MNIST Dataset Exploration and Model Performance Evaluation with Power BI and Python

This project explores the MNIST dataset and evaluates the performance of machine learning models using Power BI and Python scripting. The MNIST dataset, a well-known benchmark for image recognition, consists of 70,000 images of handwritten digits (0–9) that are widely used for training and evaluating image classification algorithms.

In this project, Power BI’s data visualization capabilities are combined with Python’s data processing and machine learning libraries to perform an interactive exploration of the dataset and assess the performance of machine learning models, specifically a Multi-Layer Perceptron (MLP) and a Support Vector Machine (SVM) with RBF Kernel.

THis project connects Power BI and Python to load MNIST sample dataframe by using a simple python script. Here is how that is done:

import pandas as pd
import numpy as np
from keras.datasets import mnist

# Load the MNIST dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Set the sample size
sample_size = 1000  # Adjust this number as needed

# Sample the training and testing data
train_indices = np.random.choice(train_X.shape[0], sample_size, replace=False)
test_indices = np.random.choice(test_X.shape[0], sample_size, replace=False)

# Select the samples for both training and test sets
train_X_sample = train_X[train_indices]
train_y_sample = train_y[train_indices]
test_X_sample = test_X[test_indices]
test_y_sample = test_y[test_indices]

# Reshape the images to have 784 pixel columns and add the label column
train_df = pd.concat([pd.DataFrame(train_X_sample.reshape(sample_size, -1)), 
                      pd.Series(train_y_sample, name="label")], axis=1)
test_df = pd.concat([pd.DataFrame(test_X_sample.reshape(sample_size, -1)), 
                     pd.Series(test_y_sample, name="label")], axis=1)

# Rename columns to 'pixel_0', 'pixel_1', ..., 'pixel_783', 'label'
train_df.columns = [f'pixel_{i}' for i in range(784)] + ['label']
test_df.columns = [f'pixel_{i}' for i in range(784)] + ['label']

# Now you have train_df and test_df with 1000 randomly selected samples, each with 785 columns

Code Explanation
1.	Import Libraries:
python
Copy code
import pandas as pd
import numpy as np
from keras.datasets import mnist
o	pandas is used for data manipulation, particularly to create DataFrames that are easy to work with in tabular form.
o	numpy is used for numerical operations, including random sampling.
o	mnist dataset from keras.datasets provides a preloaded version of the popular MNIST dataset (handwritten digits).

2.	Load the MNIST Dataset:
python
Copy code
(train_X, train_y), (test_X, test_y) = mnist.load_data()
o	This line loads the MNIST dataset into two tuples:
	train_X and train_y contain the training images and their corresponding labels.
	test_X and test_y contain the test images and their labels.
o	train_X and test_X are arrays of shape (60000, 28, 28) and (10000, 28, 28) respectively, meaning there are 60,000 training images and 10,000 test images, each of size 28x28 pixels.

3.	Set the Sample Size:
python
Copy code
sample_size = 1000
o	sample_size is set to 1,000, meaning we want a random sample of 1,000 images from both the training and testing sets.

4.	Random Sampling of Training and Testing Data:
python
Copy code
train_indices = np.random.choice(train_X.shape[0], sample_size, replace=False)
test_indices = np.random.choice(test_X.shape[0], sample_size, replace=False)
o	np.random.choice is used to randomly select 1,000 unique indices from each set (train_X and test_X).
o	Setting replace=False ensures that the same image isn’t selected more than once.

5.	Select Samples Based on Random Indices:
train_X_sample = train_X[train_indices]
train_y_sample = train_y[train_indices]
test_X_sample = test_X[test_indices]
test_y_sample = test_y[test_indices]
o	This step extracts the randomly selected 1,000 images and labels for both training and testing sets, resulting in:
	train_X_sample: 1,000 sampled training images.
	train_y_sample: Corresponding labels for the sampled training images.
	test_X_sample: 1,000 sampled testing images.
	test_y_sample: Corresponding labels for the sampled testing images.

6.	Reshape the Images and Combine with Labels:
train_df = pd.concat([pd.DataFrame(train_X_sample.reshape(sample_size, -1)), 
                      pd.Series(train_y_sample, name="label")], axis=1)
test_df = pd.concat([pd.DataFrame(test_X_sample.reshape(sample_size, -1)), 
                     pd.Series(test_y_sample, name="label")], axis=1)
o	The images are reshaped from 28x28 to 784 columns, converting each 28x28 image into a 1D array of 784 pixels.
o	pd.DataFrame(train_X_sample.reshape(sample_size, -1)) creates a DataFrame where each row corresponds to a sampled image and has 784 columns.
o	pd.Series(train_y_sample, name="label") creates a Series for the labels with the column name label.
o	pd.concat(..., axis=1) combines the reshaped images and the label Series side-by-side, resulting in train_df and test_df with 785 columns each.

7.	Rename Columns for Clarity:

train_df.columns = [f'pixel_{i}' for i in range(784)] + ['label']
test_df.columns = [f'pixel_{i}' for i in range(784)] + ['label']

o	This renames the 784 pixel columns as pixel_0, pixel_1, ..., pixel_783, and keeps the last column named label.
o	train_df and test_df now each have 785 columns in total, with label as the last column.

Final Output
•	train_df and test_df are DataFrames with 1,000 rows (images) each, and 785 columns (784 pixel values and 1 label).
•	These DataFrames are ready for use in Power BI or other analytics tools where each image is represented as a row, with pixel data and a corresponding label for the digit.


Steps to Implement in Power BI:
1.	Open Power BI and go to Home > Get Data > Python script.
2.	Write python code in the Python script editor.
3.	Run the script. When done writing the code, hit OK. Power BI will execute the script and 
4.	Choose Tables: After the code is executed, choose the tables you need and load them. In our case, we choose both train_df or test_df tables and load them.


As stated above, the MNIST dataset contains 70,000 images (60,000 for training and 10,000 for testing), each of which is a 28x28 pixel grayscale image. These images are handwritten digits from (0-9). When flattened for analysis, each image has 784 features, which translates to a DataFrame with 784 columns.  

Important Note:
In Power BI, using a sample of the MNIST dataset (e.g., 1,000 images instead of all 70,000) is practical because:
1.	Performance and Responsiveness: Loading and processing the full dataset is slow and resource-intensive. Sampling keeps Power BI responsive, enabling quicker visualizations and analysis.
2.	Memory Constraints: Power BI has memory limitations; sampling reduces load, preventing crashes and lag.
3.	Fast Prototyping and Testing: A sample lets you quickly test visuals and insights. Once set, you can scale to larger data if needed.
This approach lets you explore and validate the dataset efficiently without overloading Power BI.
