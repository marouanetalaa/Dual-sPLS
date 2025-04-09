import unittest
import numpy as np
from dual_spls.calval import d_spls_calval
from dual_spls.cv import d_spls_cv, d_spls_errorcv
from dual_spls.listecal import d_spls_listecal
from dual_spls.metric import mse, r2_score, d_spls_metric
from dual_spls.pls import d_spls_pls
from dual_spls.predict import d_spls_predict
from dual_spls.print_model import d_spls_print
from dual_spls.ridge import d_spls_ridge
from dual_spls.split import d_spls_split
from dual_spls.type import d_spls_type
from dual_spls.lasso import d_spls_lasso
from dual_spls.elasticnet import d_spls_elasticnet

# For capturing printed output from d_spls_print
import io
import contextlib


class TestCalval(unittest.TestCase):
    def setUp(self):
        # Create a simple data matrix and response vector.
        self.X = np.random.rand(50, 5)
        self.y = np.random.rand(50)

    def test_calval_output(self):
        # Call with pcal provided and without Datatype
        result = d_spls_calval(self.X, pcal=70, y=self.y, ncells=3)
        # Check that the expected keys exist.
        self.assertIn('indcal', result)
        self.assertIn('indval', result)
        # Ensure that all indices (calibration + validation) cover the whole set.
        indcal = np.array(result['indcal'])
        indval = np.array(result['indval'])
        all_indices = np.sort(np.concatenate([indcal, indval]))
        np.testing.assert_array_equal(all_indices, np.arange(self.X.shape[0]))


class TestCV(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(30, 4)
        self.y = np.random.rand(30)

    def test_cv_optimal_ncomp(self):
        # Run cross validation for dspls "lasso"
        opt_ncomp = d_spls_cv(self.X, self.y, ncomp=5, dspls="lasso", ppnu=0.5, nu2=0.0)
        # Check that the returned value is an integer between 1 and 5.
        self.assertIsInstance(opt_ncomp, int)
        self.assertGreaterEqual(opt_ncomp, 1)
        self.assertLessEqual(opt_ncomp, 5)

    def test_errorcv_function(self):
        # Create a dummy CV split and test the errorcv function.
        cv_idx = np.random.choice(self.X.shape[0], 20, replace=False)
        errors = d_spls_errorcv(cv_idx, self.X, self.y, ncomp=3, dspls="lasso", ppnu=0.5, nu2=0.0)
        self.assertEqual(len(errors), 3)
        # Errors should be non-negative.
        self.assertTrue(np.all(errors >= 0))


class TestListecal(unittest.TestCase):
    def test_listecal_output(self):
        # For a given Datatype array and 50% selection,
        # group 1 (2 elements) should yield floor(0.5*2) = 1 and group 2 (3 elements) floor(0.5*3) = 1.
        Datatype = np.array([1, 1, 2, 2, 2])
        Listecal = d_spls_listecal(Datatype, 50)
        self.assertEqual(len(Listecal), len(np.unique(Datatype)))
        self.assertEqual(Listecal, [1, 1])


class TestMetric(unittest.TestCase):
    def test_metrics(self):
        # Create a dummy model and data.
        X = np.array([[1, 2],
                      [3, 4],
                      [5, 6]])
        # Build a model with one component that exactly predicts y.
        Bhat = np.array([[0.5],
                         [1.0]])
        intercept = np.array([0.0])
        model = {'Bhat': Bhat, 'intercept': intercept}
        # Predicted values: X @ Bhat
        y_pred = X @ Bhat + intercept
        # Let y_true be exactly these predicted values.
        y_true = np.array([[2.5],[ 5.5], [8.5]])
        mse_val = mse(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        # With perfect prediction, mse should be ~0 and r2 should be 1.
        self.assertAlmostEqual(mse_val, 0.0, places=6)
        self.assertAlmostEqual(r2, 1.0, places=6)

        metrics = d_spls_metric(model, X, y_true)
        self.assertIn('MSE', metrics)
        self.assertIn('R2', metrics)


class TestPLS(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(40, 3)
        self.y = np.random.rand(40)

    def test_pls_output(self):
        model = d_spls_pls(self.X, self.y, ncp=2, verbose=False)
        self.assertIn('scores', model)
        self.assertIn('loadings', model)
        self.assertIn('Bhat', model)
        self.assertIn('intercept', model)
        self.assertEqual(model['scores'].shape, (self.X.shape[0], 2))
        self.assertEqual(model['Bhat'].shape[1], 2)


class TestPredict(unittest.TestCase):
    def test_predict_shape(self):
        # Create a dummy model with two components and four features.
        model = {
            'Bhat': np.array([[0.2, 0.1],
                              [0.3, 0.4],
                              [0.0, 0.5],
                              [0.6, 0.2]]),
            'intercept': np.array([0.0, 0.1])
        }
        X_new = np.random.rand(10, 4)
        preds = d_spls_predict(model, X_new)
        # Expected shape is (10, 2)
        self.assertEqual(preds.shape, (10, 2))


class TestPrintModel(unittest.TestCase):
    def test_print_output(self):
        model = {
            'Bhat': np.array([[0.2, 0.1],
                              [0.0, 0.3]]),
            'intercept': np.array([0.0, 0.2]),
            'type': 'test'
        }
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            d_spls_print(model)
        output = f.getvalue()
        self.assertIn("Dual-SPLS Model Summary:", output)
        self.assertIn("Component", output)


class TestRidge(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(30, 5)
        self.y = np.random.rand(30)

    def test_ridge_output(self):
        model = d_spls_ridge(self.X, self.y, ncp=2, ppnu=0.3, verbose=False)
        self.assertIn('scores', model)
        self.assertEqual(model['scores'].shape, (self.X.shape[0], 2))
        self.assertEqual(model['Bhat'].shape[1], 2)


class TestSplit(unittest.TestCase):
    def test_split_output(self):
        # Create a simple dataset with two groups: first 10 observations are group 1 and next 10 are group 2.
        X = np.random.rand(20, 3)
        Xtype = np.array([1] * 10 + [2] * 10)
        Listecal = [3, 4]  # choose 3 from group 1 and 4 from group 2.
        indices = d_spls_split(X, Xtype, Listecal)
        self.assertEqual(len(indices), sum(Listecal))
        # For each group, check that the number of selected indices matches Listecal.
        for group, ncal in zip(np.unique(Xtype), Listecal):
            group_indices = [i for i in indices if Xtype[i] == group]
            self.assertEqual(len(group_indices), ncal)


class TestType(unittest.TestCase):
    def test_type_assignment(self):
        # Provide a simple sorted response vector.
        y = np.array([10, 20, 30, 40, 50, 60])
        # With ncells=3, we expect roughly two observations per group.
        types = d_spls_type(y, ncells=3)
        self.assertEqual(len(types), 6)
        # Since y is already sorted, we expect types to be [1, 1, 2, 2, 3, 3].
        np.testing.assert_array_equal(types, np.array([1, 1, 2, 2, 3, 3]))

class TestLasso(unittest.TestCase):
    def test_correct_dimensions(self):
        # Create a dummy data set with n=100 samples and p=10 features.
        n, p = 100, 10
        X = np.random.rand(n, p)
        # Create a response vector with 100 elements.
        y = np.random.rand(n)
        
        # Choose 3 components and a sparsity parameter ppnu (e.g., 0.5)
        ncp = 3
        ppnu = 0.5
        
        # Call the dual-sPLS LASSO function
        model = d_spls_lasso(X, y, ncp=ncp, ppnu=ppnu, verbose=False)
        
        # Check that the returned model dimensions are consistent.
        self.assertEqual(model['Bhat'].shape, (p, ncp))
        self.assertEqual(model['scores'].shape, (n, ncp))
        self.assertEqual(model['fitted_values'].shape, (n, ncp))
        # Also check that the intercept vector has length ncp.
        self.assertEqual(model['intercept'].shape[0], ncp)

    def test_dimension_mismatch(self):
        # Create X with n=100 samples and p=10 features.
        X = np.random.rand(100, 10)
        # Create a response vector y with mismatched dimensions:
        # For instance, y is provided as a 1x10 array (only one sample).
        y_wrong = np.random.rand(1, 10)
        
        # We expect a dimension mismatch error when computing Xdef.T @ yc.
        with self.assertRaises(ValueError):
            # The function should fail because y_wrong.ravel() will yield a vector of length 10,
            # while X has 100 rows. This mismatch should trigger a ValueError.
            _ = d_spls_lasso(X, y_wrong, ncp=3, ppnu=0.5, verbose=False)

    def test_input_as_column_vector(self):
        # Test that the function correctly handles a y that is given as a column vector.
        n, p = 50, 5
        X = np.random.rand(n, p)
        # Create y as a column vector (shape (n,1)) rather than a 1D array.
        y_col = np.random.rand(n, 1)
        
        # d_spls_lasso should internally flatten y_col.
        model = d_spls_lasso(X, y_col, ncp=2, ppnu=0.7, verbose=False)
        
        # Basic checks on returned shapes.
        self.assertEqual(model['Bhat'].shape, (p, 2))
        self.assertEqual(model['scores'].shape, (n, 2))



class TestElasticNetPLS(unittest.TestCase):
    def setUp(self):
        # Create a random dataset with 100 samples and 10 features.
        self.X = np.random.rand(100, 10)
        self.y = np.random.rand(100)
        self.ncp = 3  # number of latent components

    def test_output_dimensions(self):
        # Run the elastic net dual-sPLS
        model = d_spls_elasticnet(self.X, self.y, ncp=self.ncp, ppnu=0.9,
                                  en_alpha=0.5, en_ridge=0.1, verbose=False)
        self.assertEqual(model['Bhat'].shape, (self.X.shape[1], self.ncp))
        self.assertEqual(model['scores'].shape, (self.X.shape[0], self.ncp))
        self.assertEqual(len(model['intercept']), self.ncp)

    def test_input_column_vector(self):
        # Test that function correctly handles y as a column vector.
        y_col = self.y.reshape(-1,1)
        model = d_spls_elasticnet(self.X, y_col, ncp=self.ncp, ppnu=0.9,
                                  en_alpha=0.7, en_ridge=0.2, verbose=False)
        self.assertEqual(model['Bhat'].shape, (self.X.shape[1], self.ncp))
    
    def test_dimension_mismatch_error(self):
        # Supply a y with wrong length to expect an error in matmul.
        y_wrong = np.random.rand(5)  # Incorrect length; should be 100.
        with self.assertRaises(ValueError):
            _ = d_spls_elasticnet(self.X, y_wrong, ncp=self.ncp, ppnu=0.9,
                                  en_alpha=0.5, en_ridge=0.1, verbose=False)


if __name__ == '__main__':
    unittest.main()
