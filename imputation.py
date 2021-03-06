from __future__ import division, print_function

from collections import namedtuple
import numpy as np
import pandas as pd
from scipy.stats import mode, t
import statsmodels.api as sm

from rr import *


CATEGORICAL_DTYPES = ['object', 'category']

# TODO:
#   - Test categorical data analysis
#   - Add plotting features
#   - Unit tests


class Imputer(object):
    """Imputer class for multiple imputation using fully conditional specification

    Parameters
    ----------
    M : int
        Number of multiple imputations

    max_its : int (default = 10)
        Maximum number of iterations for each imputation round

    imputers : dict
        Imputation methods for imputing missing data. Dictionary should contain keys 'continuous' and 'categorical'
        along with instantiated models for each methods

    missing_values : list
        List of missing values

    initial_fill : str (default = 'random')
        Initial fill of missing values. Valid arguments are 'random' (random sampling from marginal distribution),
        'mean' (mean imputation), or 'median' (median imputation). Note, for categorical data, the default imputation
        for 'mean' or 'median' is modal imputation

    Returns
    -------
    self : object
        Instance of Imputer class
    """
    def __init__(self, M = None, max_its = 10, imputers = None, missing_values = None, initial_fill = 'random',
                 alpha = .05, verbose = False):
        
        # Define attributes and data structures
        _valid_initial_fill = ['random', 'mean', 'median']

        if not isinstance(imputers, dict):
            raise ValueError('imputers (%s) should be a dictionary data structure' % type(imputers))
        else:
            self.imputers = imputers

        if not isinstance(missing_values, list):
            self.missing_values = [missing_values]
        else:
            self.missing_values = missing_values

        if initial_fill not in _valid_initial_fill:
            raise ValueError('% not a valid argument for initial_fill. Valid arguments are %s' 
                                                    (initial_fill, _valid_initial_fill))
        else:
            self.initial_fill = initial_fill

        if alpha <= 0:
            raise ValueError("alpha (%.2f) should be greater than 0" % alpha)
        elif alpha >= 1:
            raise ValueError("alpha (%.2f) should be less than 1" % alpha)
        else:
            self.alpha = alpha

        self.M = M
        self.max_its = max_its
        self.missing_summary = {}
        self.verbose = verbose
        self.RowInfo = namedtuple('RowInfo', 'o_id m_id')


    def _find_missing(self, data = None):
        """Preliminary analysis for missing data

        Parameters
        ----------
        data : pandas DataFrame
            Data frame with missing values

        Returns
        -------
        None
            Defines attribute variables useful for subsequent analysis
        """
        names = data.columns

        # Calculate indicator matrix defined as 1 = missing, 0 otherwise
        R = np.zeros(data.shape, dtype = 'bool')
        for value in self.missing_values:
            R += (data == value)

        # Create dictionary zipping column names and column sums
        col_sums = np.sum(R.astype('float'), axis = 0)
        vs = dict(zip(names, col_sums))

        # If number of imputations not specified, use crude estimate to choose M
        if not self.M:
            self.M = int(np.max((col_sums/R.shape[0])*100))

        # Create visit sequence by ordering dictionary key/value pairs
        self.missing_summary['visit_sequence'] = [k for (k, v) in sorted(vs.iteritems(), 
                                                                         key = lambda (k, v): v) if v > 0]

        # Find missing/observed rows for columns with missing data
        for j in self.missing_summary['visit_sequence']:
            R = np.zeros(data[j].shape, dtype = 'bool')
            for value in missing_values:
                R += (data[j] == value)
            row_sums = np.sum(R.astype('float'))
            self.missing_summary[j] = self.RowInfo(o_id = np.where(R == 0)[0], m_id = np.where(R > 0)[0])


    def _single_impute(self, data = None, stat = None):
        """Unconditional single imputation method using descriptive statistics such as mean and median

        Parameters
        ----------
        data : pandas DataFrame
            Data frame with missing values

        stat : str
            Method for single imputation. Valid arguments are 'mean' or 'median'

        Returns
        -------
        data_filled : pandas DataFrame
            Data frame with imputed values using single imputation method
        """
        # Iterate over missing columns and fill with mode() or stat() for categorical and continuous data, respectively.
        data_filled = data.copy()
        for name in self.missing_summary['visit_sequence']:
            if str(data_filled.dtypes[name]) in CATEGORICAL_DTYPES:
                data_filled[name].ix[self.missing_summary[name].m_id] = \
                                        mode(data_filled[name].ix[self.missing_summary[name].o_id])[0]
            else:
               data_filled[name].ix[self.missing_summary[name].m_id] = \
                                        stat(data_filled[name].ix[self.missing_summary[name].o_id])

        return data_filled


    def _random_impute(self, data = None):
        """Randomly fill missing values from draws from observed marginal distributions

        Parameters
        ----------
        data : pandas DataFrame
            Data frame with missing values

        Returns
        -------
        data_filled : pandas DataFrame
            Data frame with imputed values using random draws from marginal distributions
        """
        # Iterate over missing columns and fill missing values with random draws from marginal distributions
        data_filled = data.copy()  
        for name in self.missing_summary['visit_sequence']:
            n_mis = data_filled[name].ix[self.missing_summary[name].m_id].shape[0] 
            data_filled[name].ix[self.missing_summary[name].m_id] = \
                                    np.random.choice(data_filled[name].ix[self.missing_summary[name].o_id],
                                                     replace = True, 
                                                     size = n_mis)
        return data_filled


    def _fill_missing(self, data = None):
        """Initial imputation of missing data using simple imputation methods

        Parameters
        ----------
        data : pandas DataFrame
            Data frame with missing values

        Returns
        -------
        data_filled : pandas DataFrame
            Data frame with missing values imputed using simple imputation method
        """
        # Get missing summary information
        self._find_missing(data)

        # Initial data fill
        if self.initial_fill == 'random':
            return self._random_impute(data)
        elif self.initial_fill == 'mean':
            return self._single_impute(data, np.mean)
        else:
            return self._single_impute(data, np.median)


    def _prepare_data(self, data = None, label_col = None):
        """Prepare data for iterative imputation

        Parameters
        ----------
        data : pandas DataFrame
            Data frame with missing values

        label_col : str
            Name of dependent variable for current imputation round

        Returns
        -------
        X_obs : 2d array-like
            Array of covariates based on observed data

        X_mis : 2d array-like
            Array of covariates based on missing data

        y_obs : 1d array-like
            Array of dependent variable based on observed data
        """
        # Indices for data preparation
        self.N = data.shape[0]
        o_id = self.missing_summary[label_col].o_id
        m_id = self.missing_summary[label_col].m_id
        other_cols = [data.columns[i] for i in range(data.shape[1]) if data.columns[i] != label_col]

        # Split up data
        X_obs = data[other_cols].ix[o_id]
        X_mis = data[other_cols].ix[m_id]
        y_obs = data[label_col].ix[o_id]

        return X_obs, X_mis, y_obs


    def impute(self, data = None, func_name = None):
        """Multiple imputation using fully conditional specification

        Parameters
        ----------
        data : pandas DataFrame
            Data frame with missing value

        func_name : str 
            Name of function call for predicting missing values

        Returns
        -------
        mi_data : list
            List of imputed data sets
        """
        # Error checking
        assert(isinstance(data, pd.DataFrame)), "data is type %s, needs to be pandas DataFrame" % (type(data))
        assert(func_name), "func_name not specified, see documentation"
        if self.imputers['continuous']:
            assert(hasattr(self.imputers['continuous'], func_name)), "continuous imputer does not have %s method" % func_name
        if self.imputers['categorical']:
            assert(hasattr(self.imputers['categorical'], func_name)), "categorical imputer does not have %s method" % func_name
        
        # Start imputation scheme
        mi_data = []
        for m in range(self.M):

            if self.verbose:
                print('\nImputation %d/%d' % (m + 1, self.M))

            # Initialize imputation scheme
            counter, data_m = 0, self._fill_missing(data)

            # Begin FCS imputation
            while counter < self.max_its:

                if self.verbose:
                    print('\tIteration %d/%d' % (counter + 1, self.max_its))

                for name in self.missing_summary['visit_sequence']:

                    # Impute based on data type
                    X_obs, X_mis, y_obs = self._prepare_data(data_m, label_col = name)

                    if str(data_m.dtypes[name]) in CATEGORICAL_DTYPES:
                        self.imputers['categorical'].fit(X = X_obs, y = y_obs)
                        data_m[name].ix[self.missing_summary[name].m_id] = eval("""self.imputers['categorical'].{method}(X_mis.values)""".format(method = func_name))
                    else:
                        self.imputers['continuous'].fit(X = X_obs, y = y_obs)
                        data_m[name].ix[self.missing_summary[name].m_id] = eval("""self.imputers['continuous'].{method}(X_mis.values)""".format(method = func_name))

                counter += 1

            # Append imputed data set to list
            mi_data.append(data_m)

        return mi_data


    @staticmethod
    def apply(mi_data = None, func = None, label_col = None):
        """Applies a function handle to all imputed data sets in mi_data

        Parameters
        ----------
        mi_data : list
            List of imputed data sets

        func : function handle
            Function that takes in a pandas data frame with positional arguments (X, y),
            where X are the features or covariates and y is the label or response

        label_col : str
            Name of column used as label or response in pandas data frame

        Returns
        -------
        results : list
            List of estimates specified by return argument of function handle func()
        """
        results = []
        feature_cols = [mi_data[0].columns[i] for i in range(mi_data[0].shape[1]) if mi_data[0].columns[i] != label_col]
        for i in range(len(mi_data)):
            results.append(func(mi_data[i][feature_cols], mi_data[i][label_col]))
        return results


    @staticmethod
    def _linear_reg(X = None, y = None, fit_intercept = True):
        """Apply statsmodels linear regression to imputed data sets and return coefficients and 
           variances

        Parameters
        ----------
        X : 2d array-like
            Feature matrix

        y : 1d array-like
            Labels or response

        Returns
        -------
        """
        n = X.shape[0]
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values

        # Add vector of ones for intercept
        if fit_intercept:
            ones = np.ones((n, 1))
            X = np.hstack((ones, X))

        # Estimate model, get parameters and variances
        clf = sm.GLM(y, X, family = sm.families.Gaussian())
        params = clf.fit().params
        variances = np.diag(-np.linalg.inv(clf.information(params)))

        return (np.asarray(params), np.asarray(variances))


    def pool(self, Qstar = None, U = None, df = None):
        """Pooling phase for aggregating multiple imputation estimates

        Parameters
        ----------
        Qstar : 1d array-like
            Array of point estimates 

        U : 1d array-like
            Array of variance estimates

        df : int
            Degrees of freedom for model used in estimation

        Returns
        -------
        estimates : dict
            Dictionary of multiple imputation estimates
        """
        # Average point estimate
        Qbar = np.mean(Qstar, axis = 0)

        # Within-imputation variance, between-imputation variance, total variance estimates
        Ubar = np.mean(U, axis = 0)
        Bm = np.var(Qstar, axis = 0)
        Tm = Ubar + (1 + 1/self.M)*Bm
        
        # Relative increase in variance due to nonresponse
        r = (1 + (1/self.M))*Bm/Ubar
        
        # Unadjusted degrees of freedom
        df_unadj = (self.M - 1) * (1 + (1/r))**2

        # Adjusted degrees of freedom
        lambda_est = (Bm + Bm/self.M)/Tm
        df_adj = ((df + 1)/(df + 3))*df*(1 - lambda_est)
        
        # Fraction of missing information
        fmi = (r + 2/(df_unadj + 3))/(r + 1)

        # Confidence intervals
        se = np.sqrt(Tm)
        ll, ul = Qbar + t.ppf(self.alpha/2., df_adj)*se, Qbar + t.ppf(1 - self.alpha/2., df_adj)*se

        # Update dictionary
        estimates = {}
        estimates['point'] = Qbar
        estimates['se'] = se
        estimates['r'] = r
        estimates['df_adj'] = df_adj
        estimates['ll'] = ll
        estimates['ul'] = ul
        estimates['fmi'] = fmi

        return estimates


if __name__ == "__main__":

    ## EXAMPLE PIPELINE ##
    
    # Simulate small data set
    N, M = 50, 5
    df = N - 4
    data = pd.DataFrame(np.random.normal(0, 1, (N, 3)), columns = ['x', 'y', 'z'])
    
    # Create missing values with different indicators
    data.ix[1:7, 0] = -999
    data.ix[1:2, 1] = -777
    data.ix[7:8, 2] = -666
    missing_values = [-666, -777, -999]

    # Define imputers for continuous and categorical variables
    classifier = RecursiveClassifier(verbose = False, min_samples_leaf = 30) # Not used for shown for example
    regressor = RecursiveRegressor(verbose = False, min_samples_leaf = 10)
    imputers = {'categorical': None, 'continuous': regressor}

    # Define imputation model
    clf = Imputer(M = M, 
                  max_its = 10, 
                  verbose = True, 
                  imputers = imputers, 
                  missing_values = missing_values, 
                  initial_fill = 'random')

    # Multiply impute data
    mi_data = clf.impute(data, func_name = 'sample')

    # Apply function to each imputed data set
    mi_estimates = clf.apply(mi_data = mi_data, func = Imputer._linear_reg, label_col = 'y')

    # Pool results to obtain multiply imputed estimates
    pooled_estimates = clf.pool(Qstar = [mi_estimates[i][0] for i in range(M)], 
                                U = [mi_estimates[i][1] for i in range(M)], 
                                df = df)

    # Display results (lazy formatting!)
    print('\n')
    print('{:<10}{:^40}'.format('Estimate', 'Value'))
    print('-'*50)
    for key, value in pooled_estimates.iteritems():
        print('{:<10}{:^40}'.format(key, value))
