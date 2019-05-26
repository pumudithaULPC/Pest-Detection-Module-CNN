# Pest-Detection-Module-CNN
This is a multi class classifier CNN based on Tensorflow, Keras and Python. This Module is capable of identifying Insects(or any other type of objects) 
and classify them accordingly.

Steps To train this CNN with your own data:



1. Add the train images to a directory and classify them by adding different classes in different folders.
2.  In the getdata.py file, modify DATADIR with the path to your training images directory. Also add the names of the classes to CATEGORIES and 
  modify. 
3.  In the cnnbuild.py file modify the "num of classes" variable with the number of classes that are required to train the model.
  You can also change  other important variables like the number of epochs, optimiser and etc.
 4. Add the batch of test data into test.py and replace the CATEGORIES with the name of the classes you added in the 1st step. 
  also modify the path_in_str_in_str with the path to the test data directory. 
 5. Now run the project in the following order.
 
  1.get_data.py 
  2.resize_data.py
  3.building_training_data.py
  4.pickle_builder.py
  5.cnn_build.py
  6.test.py
