from __future__ import division, print_function

from collections import namedtuple
import numpy as np
import pandas as pd
from scipy.stats import mode

from rr import *


CATEGORICAL_DTYPES = ['object', 'category']

# TODO:
#   - Reinstantiate models during imputation iterations
#   - Test categorical data analysis
#   - Add plotting features
#   - Add functionality to apply models to all data sets


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
                 verbose = False):
        
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
        o_id = self.missing_summary[label_col].o_id
        m_id = self.missing_summary[label_col].m_id
        other_cols = [data.columns[i] for i in range(data.shape[1]) if data.columns[i] != label_col]

        # Split up data
        X_obs = data[other_cols].ix[o_id]
        X_mis = data[other_cols].ix[m_id]
        y_obs = data[label_col].ix[o_id]

        return X_obs, X_mis, y_obs


    def impute(self, data = None):
        """Multiple imputation using fully conditional specification

        Parameters
        ----------
        data : pandas DataFrame
            Data frame with missing value

        Returns
        -------
        mi_data : list
            List of imputed data sets
        """
        assert(isinstance(data, pd.DataFrame)), "data is type %s, needs to be pandas DataFrame" % (type(data))
        
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
                        data_m[name].ix[self.missing_summary[name].m_id] = self.imputers['categorical'].sample(X_mis.values)
                    else:
                        self.imputers['continuous'].fit(X = X_obs, y = y_obs)
                        data_m[name].ix[self.missing_summary[name].m_id] = self.imputers['continuous'].sample(X_mis.values)

                counter += 1

            # Append imputed data set to list
            mi_data.append(data_m)

        return mi_data


    def apply(self, mi_data = None, func = None, label_col = None):
        """Applies func() to all imputed data sets in mi_data

        Parameters
        ----------
        mi_data : list
            List of imputed data sets

        func : function
            Function that takes in a pandas data frame

        Returns
        -------
        """
        results = []
        other_cols = [mi_data[0].columns[i] for i in range(mi_data[0].shape[1]) if mi_data[0].columns[i] != label_col]
        for i in range(len(mi_data)):
            results.append(func(mi_data[i][other_cols], mi_data[i][label_col]))
        return results


if __name__ == "__main__":
    data = pd.DataFrame(np.random.normal(0, 1, (50, 3)), columns = ['x', 'y', 'z'])
    data.ix[1:7, 0] = -999
    data.ix[1:2, 1] = -777
    data.ix[7:8, 2] = -666
    missing_values = [-666, -777, -999]

    classifier = RecursiveClassifier(verbose = False, min_samples_leaf = 30)
    regressor = RecursiveRegressor(verbose = False, min_samples_leaf = 10)
    imputers = {'categorical': None, 'continuous': regressor}

    clf = Imputer(M = 20, max_its = 10, verbose = True, imputers = imputers, missing_values = missing_values, initial_fill = 'median')
    clf.impute(data)
