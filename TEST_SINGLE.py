from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

num_periods = 10

num_learning_rate = 0.001
num_steps=3000
num_batch_size=1

#feature_str = "math_SAT"
#feature_str = "verb_SAT"
#feature_str = "comp_GPA"
feature_str = "high_GPA"
target_str = "univ_GPA"


def preprocess_features(grade):
  """Prepares input features from California housing data set.

  Args:
    grade: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  #selected_features = grade[
  #  ["high_GPA",
  #   "math_SAT",
  #   "verb_SAT",
  #   "comp_GPA"]]
  
  selected_features = grade[
    [feature_str]]
  
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  #processed_features["rooms_per_person"] = (
  #  grade["total_rooms"] /
  #  grade["population"])
  return processed_features

def preprocess_targets(grade):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    grade: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets[target_str] = (
    grade[target_str])
  return output_targets

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of multiple features.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(1000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])


def train_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model of multiple features.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `grade` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `grade` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `grade` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `grade` to use as target for validation.
      
  Returns:
    A `LinearRegressor` object trained on the training data.
  """
  periods = num_periods
  steps_per_period = steps / periods
  
  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer
  )
  
  # Create input functions.
  training_input_fn = lambda: my_input_fn(
      training_examples, 
      training_targets[target_str], 
      batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(
      training_examples, 
      training_targets[target_str], 
      num_epochs=1, 
      shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(
      validation_examples, validation_targets[target_str], 
      num_epochs=1, 
      shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period,
    )
    # Take a break and compute predictions.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
        
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %03d : %0.3f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()

  return linear_regressor

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#grade = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

grade = pd.read_csv("./gpa.csv", sep=",")

grade = grade.reindex(
     np.random.permutation(grade.index))

grade = grade.reindex(
     np.random.permutation(grade.index))

grade["math_SAT"] /= 100.0
grade["verb_SAT"] /= 100.0

training_examples = preprocess_features(grade.head(70))
print(training_examples.describe())

training_targets = preprocess_targets(grade.head(70))
training_targets.describe()

validation_examples = preprocess_features(grade[75:90])
print(validation_examples.describe())

validation_targets = preprocess_targets(grade[75:90])
validation_targets.describe()

linear_regressor = train_model(
    learning_rate=num_learning_rate,
    steps=num_steps,
    batch_size=num_batch_size,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

test_examples = preprocess_features(grade.tail(15))
test_targets = preprocess_targets(grade.tail(15))

predict_test_input_fn = lambda: my_input_fn(
      test_examples, 
      test_targets[target_str], 
      num_epochs=1, 
      shuffle=False)

test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item['predictions'][0] for item in test_predictions])

root_mean_squared_error = math.sqrt(
    metrics.mean_squared_error(test_predictions, test_targets))

min_value = grade[target_str].min()
max_value = grade[target_str].max()
min_max_difference = max_value - min_value

print("Min. "+target_str+" Value: %0.3f" % min_value)
print("Max. "+target_str+" Value: %0.3f" % max_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

#calibration_data = pd.DataFrame()
#calibration_data["predictions"] = pd.Series(test_predictions)
#calibration_data["targets"] = pd.Series(test_targets)
#print(calibration_data.describe())

sample = grade.tail(15)

# Get the min and max total_rooms values.
x_0 = sample[feature_str].min()
x_1 = sample[feature_str].max()
# Retrieve the final weight and bias generated during training.
weight = linear_regressor.get_variable_value('linear/linear_model/'+feature_str+'/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
# Get the predicted median_house_values for the min and max total_rooms values.
y_0 = weight * x_0 + bias 
y_1 = weight * x_1 + bias
# Plot our regression line from (x_0, y_0) to (x_1, y_1).
plt.plot([x_0, x_1], [y_0, y_1], c='r')
# Label the graph axes.
plt.ylabel(target_str)
plt.xlabel(feature_str)
# Plot a scatter plot from our data sample.
plt.scatter(sample[feature_str], sample[target_str])
# Display graph.
plt.show()

