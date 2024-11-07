"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""
import io
import unittest
import numpy as np
from regression import (
			logreg,
			utils)

def test_updates():
	# Check that your gradient is being calculated correctly
	# What is a reasonable gradient? Is it exploding? Is it vanishing? 
	
	# Check that your loss function is correct and that 
	# you have reasonable losses at the end of training
	# What is a reasonable loss?

	pass

   

def test_predict():
	# Check that self.W is being updated as expected 
 	# and produces reasonable estimates for NSCLC classification
	# What should the output should look like for a binary classification task?

	# Check accuracy of model after training

	pass


def test_sigmoid_output_range(self):
        """
        Test that the sigmoid function returns values between 0 and 1.
        """
        # Test inputs: a range of positive, negative, and zero values
        test_values = np.array([-1000, -10, -1, 0, 1, 10, 1000])
        
        # Call the sigmoid function
        sigmoid_outputs = self.model.sigmoid(test_values)
        
        # Check that all outputs are within the (0, 1) range
        self.assertTrue(np.all(sigmoid_outputs > 0) & np.all(sigmoid_outputs < 1),
                        "Sigmoid function output is not in the range (0, 1) for extreme values.")
        




"""

        - Check if fit appropriately trains model & weights get updated
        - Check loss approaches 0 
        - Check predict works as intended


#because using a binary probability approach, the smaller the loss value the better
# Test cases for gradient (2)
	Positive test cases
		convergance 
		
	Negative
		Incorrect input - maybe didn't read the data well
		non-binary labels

# Test cases for loss (2)
	Positive tests
		only 1, positive label (classification of data)
	
	Negative tests
		non-binary labels 
		Lack of "sigmodial curve"

		
"""