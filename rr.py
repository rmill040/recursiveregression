from __future__ import division, print_function

from collections import namedtuple, OrderedDict
import numpy as np
import pandas as pd
import patsy
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.spatial.distance import cdist
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import warnings

import pdb


__all__ = ["BaseRecursiveModel", "RecursiveRegressor", "RecursiveClassifier"]


class BaseRecursiveModel(object):
    """Base model for recursive regression modeling

    Parameters
    ----------
    min_samples_leaf : int
        Minimum sample size in a terminal node for tree splitting

    splitter : str (default = 'best')
        Method for tree splitting. Valid arguments are 'best' and 'random'

    fit_intercept : bool (default = True)
        Whether to fit intercept in linear models

    verbose : bool (default = True)
        Whether to print status of fitting procedure

    Returns
    -------
    self : object
        Instance of BaseRecursiveModel
    """
    def __init__(self, min_samples_leaf = None, splitter = 'best', fit_intercept = True, verbose = False):

        # Define attribute variables
        _valid_splitters, _valid_bool = ['best', 'random'], [True, 1, False, 0]
        if splitter in _valid_splitters:
            self.splitter = splitter
        else:
            raise ValueError('%s not a valid splitter. Valid arguments are %s' % (splitter, _valid_splitters))

        if fit_intercept in _valid_bool:
            self.fit_intercept = fit_intercept
        else:
            raise ValueError('%s not a valid fit_intercept. Valid arguments are %s' % (fit_intercept, _valid_bool))

        if verbose in _valid_bool:
            self.verbose = verbose
        else:
            raise ValueError('%s not a valid verbose. Valid arguments are %s' % (verbose, _valid_bool))

        if self._CLASSIFIER:
            if min_samples_leaf < 30:
                warnings.warn('min_samples_leaf too low, setting to minimum value of 30')
                self.min_samples_leaf = 30
            else:
                self.min_samples_leaf = min_samples_leaf

        else:
            if min_samples_leaf < 10:
                warnings.warn('min_samples_leaf too low, setting to minimum value of 10')
                self.min_samples_leaf = 10
            else:
                self.min_samples_leaf = min_samples_leaf

        self.level, self.converged = 0, False
        self.terminal_nodes, self.summary = [], OrderedDict()
        self.DataSplit = namedtuple('DataSplit', 'exog metric coef n')


    def _create_matrices(self, formula = None, data = None):
        """Create design matrices from patsy-type formula

        Parameters
        ----------
        formula : str
            Patsy formula

        data : pandas DataFrame
            Pandas dataframe

        Returns
        -------
        endog : 1d array-like
            Array for dependent variable

        exog : 2d array-like
            Array of covariates

        part : 2d array-like
            Array of partitioning variables
        """
        # Separate structural model from partitioning model and obtain matrices
        formula_split = formula.split('|')
        structural, partitioning = formula_split[0].strip(), formula_split[1].strip()

        # If intercept specified keep otherwise remove
        if self.fit_intercept:
            endog, exog = patsy.dmatrices(formula_like = structural, data = data)
        else:
            endog, exog = patsy.dmatrices(formula_like = structural[0] + ' ~ 0 + ' + structural.split('~')[-1].strip(), 
                                          data = data)
        part = patsy.dmatrix(formula_like = '0 + ' + partitioning, data = data)

        return endog, exog, part


    def _tree_splitter(self, endog = None, exog = None, part = None):
        """Tree splitter function

        Parameters
        ----------
        endog : 1d array-like
            Array for dependent variable

        exog : 2d array-like
            Array of covariates

        part : 2d array-like
            Array of partitioning variables

        Returns
        -------
        endog_split : list
            Split endogenous variable based on tree splitting classes

        exog_split : list
            Split exogenous variable(s) based on tree splitting classes

        part_split : list
            Split partitioning variable(s) based on tree splitting classes

        stop : bool
            Whether stopping criteria met for tree splitting
        """
        # Train decision tree and find leaf indices
        self.tree_model.fit(X = part, y = endog)
        idx = self.tree_model.apply(X = part)

        # Check for stopping criteria
        if np.all(idx == 0):
            endog_split, exog_split, part_split, stop = None, None, None, True 
        
        else:
            stop = False
            endog_split = [endog[idx == i] for i in (1, 2)]
            exog_split = [exog[idx == i] for i in (1, 2)]
            part_split = [part[idx == i] for i in (1, 2)]

            # Check if all labels are same in a node, if so node is pure
            if self._CLASSIFIER:
                for c in range(2):
                    if np.all(endog_split[c] == 0) or np.all(endog_split[c] == 1):
                    # if endog_split[c].tolist().count(endog_split[c][0]) == len(endog_split[c]):
                    # if np.allclose(endog_split[c], np.repeat(endog_split[c][0], endog_split[c].shape[0])):
                        stop = True
                        break

        return endog_split, exog_split, part_split, stop


    def _model_summary(self, endog = None, exog = None):
        """Trains linear model and returns metrics for current node

        Parameters
        ----------
        endog : 1d array-like
            Array for dependent variable

        exog : 2d array-like
            Array of covariates

        Returns
        -------
        DataSplit : namedtuple
            Summary information in current node. Contains metric, coefficients from linear fitting
            exogenous variables, and sample size
        """
        # Create linear model, train, and return metrics
        self.linear_model.fit(X = exog, y = endog.ravel())
        y_hat = self.linear_model.predict(exog)

        # Drop vector of 1s to save in data structure
        if self.fit_intercept:
            exog = exog[:, 1:]

        return self.DataSplit(metric = self._metric(y_true = endog, y_hat = y_hat, 
                                                    exog = exog, coef = self.linear_model.coef_), 
                              coef = self.linear_model.coef_, 
                              exog = exog,
                              n = len(endog))


    def _nearest_terminal_node(self, x = None, metric = 'euclidean'):
        """Use nearest neighbor (k = 1) method to find most similar terminal node to vector x

        Parameters
        ----------
        x : 1d array-like
            Test vector

        metric : str (default = 'euclidean')
            Method for calculating distances. See scipy.spatial.distances for valid options

        Returns
        -------
        nearest_node : str
            Name of nearest terminal node
        """
        # Loop over terminal nodes and calculate distance metrics
        nearest_node, smallest_dist = None, 1e10
        for name in self.terminal_nodes:
            dists = cdist(x.reshape(1, -1), self.summary[name].exog, metric = metric)
            eps = np.min(dists)
            if eps < smallest_dist:
                smallest_dist, nearest_node = eps, name
        return nearest_node


    def _total_metric(self):
        """Calculate metrics from original sample and summed terminal nodes

        Parameters
        ----------
        None

        Returns
        -------
        original_metric : float
            Metric from full sample

        terminal_nodes_metric : float
            Summed metric across terminal nodes 
        """
        original_metric = self.summary['P'].metric    
        terminal_nodes_metric = 0
        for i in self.terminal_nodes:
            terminal_nodes_metric += self.summary[i].metric
        return original_metric, terminal_nodes_metric


    def fit(self, formula = None, data = None, **kwargs):
        """Recursively fit linear models

        Parameters
        ----------
        formula : str
            Patsy formula specifying linear model
            Example: y ~ x1 + x2 | z1 + z2 --> structural model before | and partitioning model after |

        data : pandas dataframe
            Pandas DataFrame that contains labeled variables based on formula

        **kwargs : dict
            Keyword arguments used to pass features X and label y similar to standard scikit learn API.
            By default, the partitioning variables will be all columns in X

        Returns
        -------
        None
            Trained recursive linear model
        """
        # Create matrices and define number of predictors to calculate metrics later in pipeline
        if formula:
            assert(isinstance(data, pd.DataFrame)), "data is type %s, needs to be pandas DataFrame" % (type(data))
            endog, exog, part = self._create_matrices(formula = formula, data = data)
        else:
            exog, endog = kwargs['X'], kwargs['y']
            part = exog.copy()
            if self.fit_intercept:
                exog = np.hstack((np.ones((exog.shape[0], 1)), exog)) 
        self.k = exog.shape[1]

        # Begin analysis
        name,  = 'P'
        queue = [(name, endog, exog, part)]
        while queue:
            
            # Get current parent node model information 
            name, endog, exog, part = queue.pop(0)
            self.summary[name] = self._model_summary(endog = endog, exog = exog)

            if self.verbose:
                print('\n---- LEVEL %d ----' % self.level)
                print('Parent node %s: n = %d, metric = %f' % (name, self.summary[name].n, self.summary[name].metric))

            # If parent node is smaller than threshold, append as terminal node and continue
            if self.summary[name].n < self.min_samples_leaf:
                self.terminal_nodes.append(name)
                if self.verbose:
                    print('\tTERMINAL -- TOO SMALL')
                continue

            # Get tree split based on partitioning variables
            endog_split, exog_split, part_split, stop = self._tree_splitter(endog = endog, exog = exog, part = part) 

            # If stopping criteria not met in tree splitting continue with analysis          
            if not stop:
                info_list, child_metrics, child_n = [], np.zeros(2), np.zeros(2)

                # Loop over child nodes and calculate information for each node
                for c in range(2):
                    info_list.append(self._model_summary(endog = endog_split[c], exog = exog_split[c]))
                    child_metrics[c], child_n[c] = info_list[c].metric, info_list[c].n

                # Aggregate metric across children
                child_sum = np.sum(child_metrics)

                if self.verbose:
                    print('Child nodes: n = (%d, %d), metrics = (%f, %f)' % (child_n[0], child_n[1], child_metrics[0], child_metrics[1]))
                    print('\tTotal metric: %f' % child_sum)                

                # Check metric threshold and see if split makes sense
                if np.sum(child_sum) < self.summary[name].metric:
                    for c in range(2):
                        key = name + str(c)
                        self.summary[key] = info_list[c]

                        # If size child node greater than threshold continue with analysis, else node is terminal
                        if child_n[c] > self.min_samples_leaf:
                            queue.append((key, endog_split[c], exog_split[c], part_split[c]))
                        
                        else:
                            if self.verbose:
                                print('\tChild node %s TERMINAL -- TOO SMALL' % key)
                            self.terminal_nodes.append(key)

                # Split hurts metric so consider both children as terminal nodes
                else:
                    for c in range(2):
                        key = name + str(c)
                        if self.verbose:
                            print('\tChild node %s TERMINAL -- LOSS INCREASED' % key)
                        self.terminal_nodes.append(key)

            # Stopping criterion met in tree splitting so all labels are the same
            else:
                self.terminal_nodes.append(name)
                if self.verbose:
                    print('\tTERMINAL -- TREE STOPPING')

            # Continue with partitioning
            self.level += 1

        # Convergence is True if the routine finishes
        self.converged = True


    def predict(self, X = None, metric = 'euclidean'):
        """Predict labels based on linear model in closest terminal nodes

        Parameters
        ----------
        X : 2d array-like
            Array of test features

        metric : str (default = 'euclidean')
            Method for calculating distances. See scipy.spatial.distances for valid options

        Returns
        -------
        y_hat : 1d array-like
            Array of predicted labels
        """
        assert(self.converged == True), 'Train model before running predict() method'
        
        # Loop through test vectors and make predictions
        n = X.shape[0]
        y_hat = np.zeros(n)
        for i in xrange(n):
            nearest_node = self._nearest_terminal_node(x = X[i, :], metric = metric)
            y_hat[i] = self._predict_y(x = X[i, :], terminal_node = nearest_node)

        return y_hat


class RecursiveClassifier(BaseRecursiveModel):
    """Recursive linear classifier class. Inherits BaseRecursiveModel class

    Parameters
    ----------
    criterion : str (default = 'gini')
        Criterion for evaluating tree splitting model. Valid arguments are 'gini' (gini index) 
        and 'entropy' (information gain)

    Returns
    -------
    self : object
        Instance of RecursiveClassifier class
    """
    def __init__(self, criterion = 'gini', *args, **kwargs):
        
        # Define attribute variables
        self._CLASSIFIER = True
        super(RecursiveClassifier, self).__init__(*args, **kwargs)
        _valid_criterion = ['gini', 'entropy']
        if criterion in criterion:
            self.criterion = criterion
        else:
            raise ValueError('%s not a valid criterion. Valid arguments are %s' % (criterion, _valid_criterion))

        self.linear_model = LogisticRegression(fit_intercept = False) 
        self.tree_model = DecisionTreeClassifier(min_samples_leaf = self.min_samples_leaf,
                                                 max_depth = 1,
                                                 splitter = self.splitter,
                                                 criterion = self.criterion)


    @staticmethod
    def _logit(exog = None, coef = None):
        """Logit transformation that generates predicted probabilities based on exog and coef

        Parameters
        ----------
        exog : 2d array-like
            Array of covariates

        coef : 1d array-like
            Array of coefficients

        Returns
        -------
        p : 1d array-like
            Predicted probabilities
        """
        exp_Xb = np.exp(np.dot(exog, coef.reshape(-1, 1)))
        return (exp_Xb / (1 + exp_Xb))


    def _metric(self, y_true = None, exog = None, coef = None, **kwargs):
        """Negative log-likelihood (Bernoulli distribution) for linear classifier for n samples as 
           y*log(p) + (1 - y)*log(1 - p)

        Parameters
        ----------
        y_true : 1d array-like
            Array of ground truth labels

        exog : 2d array-like
            Array of covariates

        coef : 1d array-like
            Array of coefficients

        **kwargs : keyword arguments
            Not used - set to allow compatability with RecursiveRegressor

        Returns
        -------
        nll : float
            Metric representing negative log-likelihood in current node
        """
        # Return sum because averaging now and combining child node metrics would require re-weighting before combining
        if self.fit_intercept:
            exog = np.hstack((np.ones((exog.shape[0], 1)), exog))         
        p = self._logit(exog = exog, coef = coef)
        return -np.sum(y_true*np.log(p) + (1 - y_true)*np.log(1 - p))


    def _predict_y(self, x = None, terminal_node = None):
        """Predict label for test vector x based on fitted linear model in terminal node

        Parameters
        ----------
        x : 1d array-like
            Array of test features

        terminal_node : str
            Terminal node name

        Returns
        -------
        y_hat : int
            Predicted class label
        """
        # Grab coefficients from terminal node
        coef = self.summary[terminal_node].coef.reshape(-1, 1)

        # Calculate predicted probability and threshold to get class label
        if self.fit_intercept:
            p = self._logit(exog = np.insert(x, 0, 1).reshape(1, -1), coef = coef)
        else:
            p = self._logit(exog = x.reshape(1, -1), coef = coef)

        if p < .5:
            return 0
        else:
            return 1


    def _draw_y(self, x = None, terminal_node = None):
        """Draw predicted label for test vector x based on distribution of fitted linear model 
           in terminal node

        Parameters
        ----------
        X : 1d array-like
            Array of test features

        terminal_node : str
            Terminal node name

        Returns
        -------
        y_hat : float
            Randomly drawn value for y_hat
        """
        # Grab coefficients from terminal node
        coef = self.summary[terminal_node].coef.reshape(-1, 1)
        if self.fit_intercept:
            p = self._logit(exog = np.insert(x, 0, 1).reshape(1, -1), coef = coef)
        else:
            p = self._logit(exog = x.reshape(1, -1), coef = coef)

        # Return random draw from Bern(p)
        return np.random.binomial(1, p, 1)


    def sample(self, X = None, metric = 'euclidean'):
        """Sample labels based distribution of closest terminal nodes

        Parameters
        ----------
        X : 2d array-like
            Array of test features

        metric : str (default = 'euclidean')
            Method for calculating distances. See scipy.spatial.distances for valid options

        Returns
        -------
        y_hat : 1d array-like
            Array of predicted labels
        """
        assert(self.converged == True), 'Train model before running sample() method'

        # Loop through test vectors and draw random variables
        n = X.shape[0]
        y_hat = np.zeros(n)
        for i in xrange(n):
            nearest_node = self._nearest_terminal_node(x = X[i, :], metric = metric)
            y_hat[i] = self._draw_y(x = X[i, :], terminal_node = nearest_node)

        return y_hat  


class RecursiveRegressor(BaseRecursiveModel):
    """Recursive linear regression class. Inherits BaseRecursiveModel class

    Parameters
    ----------
    criterion : str (default = 'mse')
        Criterion for evaluating linear regression model. Valid arguments are 'mse' (mean squared error) 
        and 'mae' (mean absolute error)

    Returns
    -------
    self : object
        Instance of RecursiveRegression class
    """
    def __init__(self, criterion = 'mse', *args, **kwargs):
        
        # Define attribute variables
        self._CLASSIFIER = False
        super(RecursiveRegressor, self).__init__(*args, **kwargs)
        _valid_criterion = ['mse', 'mae']
        if criterion in criterion:
            self.criterion = criterion
        else:
            raise ValueError('%s not a valid criterion. Valid arguments are %s' % (criterion, _valid_criterion))

        self.linear_model = LinearRegression(fit_intercept = False) 
        self.tree_model = DecisionTreeRegressor(min_samples_leaf = self.min_samples_leaf,
                                                max_depth = 1,
                                                splitter = self.splitter,
                                                criterion = self.criterion)


    def _metric(self, y_true = None, y_hat =  None, **kwargs):
        """Linear regression model metrics

        Parameters
        ----------
        y_true : 1d array-like
            Array of ground truth labels

        y_hat : 1d array-like
            Array of predicted labels

        **kwargs : keyword arguments
            Not used - set to allow compatability with RecursiveClassifier 

        Returns
        -------
        metric : float
            Metric representing error in current node
        """
        # Return sum because averaging now and combining child node metrics would require re-weighting before combining
        if self.criterion == 'mse':
            return np.sum((y_true.ravel() - y_hat.ravel())**2)
        else:
            return np.sum(np.abs(y_true.ravel() - y_hat.ravel()))


    def _predict_y(self, x = None, terminal_node = None):
        """Predict label for test vector x based on fitted linear model in terminal node

        Parameters
        ----------
        x : 1d array-like
            Array of test features

        terminal_node : str
            Terminal node name

        Returns
        -------
        y_hat : float
            Predicted value
        """
        # Grab coefficients from terminal node
        coef = self.summary[terminal_node].coef.reshape(-1, 1)
        if self.fit_intercept:
            return np.dot(np.insert(x, 0, 1).reshape(1, -1), coef)
        else:
            return np.dot(x.reshape(1, -1), coef)


    def _draw_y(self, x = None, terminal_node = None):
        """Draw predicted label for test vector x based on distribution of fitted linear model 
           in terminal node

        Parameters
        ----------
        X : 1d array-like
            Array of test features

        terminal_node : str
            Terminal node name

        Returns
        -------
        y_hat : float
            Randomly drawn value for y_hat
        """
        # Calculate mean squared error in terminal node as MSE = SSE / (n - k)
        mse = self.summary[terminal_node].metric/(self.summary[terminal_node].n - self.k)

        # Grab coefficients from terminal node and calculate mean
        coef = self.summary[terminal_node].coef
        if self.fit_intercept:
            mu = np.dot(np.insert(x, 0, 1).reshape(1, -1), coef)
        else:
            mu = np.dot(x.reshape(1, -1), coef)

        # Return random draw from N(mu, mse)
        return np.random.normal(mu, mse, 1) 


    def sample(self, X = None, metric = 'euclidean'):
        """Sample labels based distribution of closest terminal nodes

        Parameters
        ----------
        X : 2d array-like
            Array of test features

        metric : str (default = 'euclidean')
            Method for calculating distances. See scipy.spatial.distances for valid options

        Returns
        -------
        y_hat : 1d array-like
            Array of predicted labels
        """
        assert(self.converged == True), 'Train model before running sample() method'
        assert(self.criterion == 'mse'), 'Criterion must be mse to use sample() method'

        # Loop through test vectors and draw random variables
        n = X.shape[0]
        y_hat = np.zeros(n)
        for i in xrange(n):
            nearest_node = self._nearest_terminal_node(x = X[i, :], metric = metric)
            y_hat[i] = self._draw_y(x = X[i, :], terminal_node = nearest_node)

        return y_hat    


if __name__ == "__main__":

    linear = False
    logistic = True

    if linear:
        data = pd.DataFrame(np.random.normal(0, 1, (1000, 8)), columns = ['y', 'x1', 'x2', 'x3', 'z1', 'z2', 'z3', 'z4'])
        rr = RecursiveRegressor(min_samples_leaf = 10, splitter = 'best', criterion = 'mse', fit_intercept = True, verbose = True)
        rr.fit('y ~ x1 + x2 + x3 | z1 + z2 + z3 + z4', data = data)
        initial, final = rr._total_metric()
        print('\nInitial Metric: %f' % initial)
        print('Final Metric: %f' % final)
        print('Nearest terminal node: %s' % rr._nearest_terminal_node(np.random.normal(0, 1, (1, 3))))

        # Generate test data
        X = np.random.normal(0, 1, (1000, 3))
        import matplotlib.pyplot as plt

        y1 = rr.predict(X = X, metric = 'euclidean')
        y2 = rr.sample(X = X, metric = 'euclidean')

        f, axarr = plt.subplots(1, 2, sharex = True, sharey = True)
        axarr[0].hist(y1)
        axarr[1].hist(y2)
        plt.show()

    if logistic:
        data = pd.DataFrame(np.random.normal(0, 1, (100, 7)), columns = ['x1', 'x2', 'x3', 'z1', 'z2', 'z3', 'z4'])
        #data['y'] = np.random.binomial(1, .5, (100, 1))
        rr = RecursiveClassifier(min_samples_leaf = 20, splitter = 'best', criterion = 'entropy', fit_intercept = True, verbose = True)
        #rr.fit('y ~ x1 + x2 + x3 | z1 + z2 + z3 + z4', data = data)
        dat = {'X': data, 'y': np.random.binomial(1, .5, (100, 1))}
        rr.fit(**dat)
        initial, final = rr._total_metric()
        print('\nInitial Metric: %f' % initial)
        print('Final Metric: %f' % final)
        print('Nearest terminal node: %s' % rr._nearest_terminal_node(np.random.normal(0, 1, (1, 7))))

        # Generate test data
        X = np.random.normal(0, 1, (100, 7))
        print(np.mean(rr.predict(X = X, metric = 'euclidean')))
        print(np.mean(rr.sample(X = X, metric = 'euclidean')))