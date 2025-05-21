import unittest
import numpy as np
import sys
from models.linearRegression import linearRegression
from models.multiRegression import multiRegression
from models.navieKMeansClustering import naiveKMeansClustering

LOG = False

# Ensure command-line arguments don't break unittest
if __name__ == "__main__":
    if len(sys.argv) > 1:
        LOG = True if sys.argv[1] == "log" else False
    sys.argv = sys.argv[:1]  # Prevent unittest from interpreting extra args



class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.X = np.array([1, 2, 3, 4, 5])
        self.y = np.array([2, 4, 6, 8, 10])
        self.model = linearRegression(self.X, self.y)
    
    def test_initialization(self):
        # Test if the model initializes correctly
        self.assertIsInstance(self.model, linearRegression)
        self.assertTrue(np.array_equal(self.model.X, self.X))
        self.assertTrue(np.array_equal(self.model.y, self.y))
        self.assertEqual(self.model.m, 0)
        self.assertEqual(self.model.b, 0)

    def test_train(self):
        # Train the model
        self.model.train(epochs=1000, alpha=0.01,log = LOG)

        # Check if parameters have been updated
        self.assertNotEqual(self.model.m, 0)
        self.assertNotEqual(self.model.b, 0)

    def test_predict(self):
        # Train the model first
        self.model.train(epochs=1000, alpha=0.01, log = LOG)
        
        # Make a prediction
        prediction = self.model.predict(6)
        self.assertIsInstance(prediction, (int, float))  # Check if the prediction is a number

        # Check the prediction is reasonably close to expected
        expected_prediction = 12
        self.assertAlmostEqual(prediction, expected_prediction, delta=0.1)  # Allow a small error

class TestMultiRegression(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.X = np.array([[1, 2],  
                        [2, 3],  
                        [3, 4],  
                        [4, 5]])  
        self.y = np.array([3, 5, 7, 9])
        self.model = multiRegression(self.X, self.y)
    
    def test_initialization(self):
        # Test if the model initializes correctly
        self.assertIsInstance(self.model, multiRegression)
        self.assertTrue(np.array_equal(self.model.X, self.X))
        self.assertTrue(np.array_equal(self.model.y, self.y))
        self.assertTrue(np.array_equal(self.model.m, np.zeros(self.X.shape[1])))
        self.assertEqual(self.model.b, 0)

    def test_train(self):
        # Train the model
        self.model.train(10000, 0.01,log=LOG)
        self.assertFalse(np.array_equal(self.model.m, np.zeros(self.X.shape[1])))
        self.assertNotEqual(self.model.b, 0)

    def test_predict(self):
        # Train the model first
        self.model.train(10000, 0.01,log=LOG)
        
        # Make a prediction
        prediction = self.model.predict(np.array([[5, 6]]))
        self.assertIsInstance(prediction, (np.ndarray))  # Check if the prediction is a number

        # Check the prediction is reasonably close to expected
        expected_prediction = 11
        self.assertAlmostEqual(prediction, expected_prediction, delta=0.1)  # Allow a small error


class TestNaiveKMeansClustering(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.data = np.array([1,2,3,11,12,13,21,22,23])
        self.model = naiveKMeansClustering(self.data, 3)
    
    def test_initialization(self):
        # Test if the model initializes correctly
        self.assertIsInstance(self.model, naiveKMeansClustering)
        self.assertTrue(np.array_equal(self.model.data, self.data))

    def test_train(self):
        # Train the model
        self.model.train(1000,log=LOG)
        self.assertFalse(np.array_equal(self.model.centroids, np.zeros(3)))

    def test_predict(self):
        # Train the model first
        self.model.train(1000, log=LOG)
        
        # Make a prediction
        prediction = self.model.predict(np.array(14))
        self.assertTrue(prediction in self.model.get_centroids())

if __name__ == '__main__':
    unittest.main()
