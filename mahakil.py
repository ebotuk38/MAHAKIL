#Bennin, K. E., Keung, J., Phannachitta, P., Monden, A., & Mensah, S. (2018). Mahakil: Diversity based oversampling approach to alleviate the class imbalance issue in software defect prediction. IEEE Transactions on Software Engineering, 44(6), 534-550.
from __future__ import division

from collections import Counter
import warnings
import numpy as np
import numpy.linalg
import pandas as pd
import math
from sklearn.utils import safe_indexing
from scipy.spatial.distance import mahalanobis
import scipy as sp

class MAHAKIL():

   def __init__(self):
     #self.sampling_type = 'over-sampling'
     self.sampling_strategy = 'auto'


   def sampling_strategy_majority(self, y, sampling_type):
    """Returns sampling target by targeting all classes but not the
      majority."""
    target_stats = Counter(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        self.sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items() if key != class_majority
        }
    else:
        raise NotImplementedError

    return self.sampling_strategy

   def sampling_strategy_dict(self, sampling_strategy, y, sampling_type):
    """Returns sampling target by converting the dictionary depending of the
    sampling."""
    target_stats = Counter(y)
    # check that all keys in sampling_strategy are also in y
    set_diff_sampling_strategy_target = (
        set(sampling_strategy.keys()) - set(target_stats.keys()))
    if len(set_diff_sampling_strategy_target) > 0:
        raise ValueError("The {} target class is/are not present in the"
                         " data.".format(set_diff_sampling_strategy_target))
    # check that there is no negative number
    if any(n_samples < 0 for n_samples in sampling_strategy.values()):
        raise ValueError("The number of samples in a class cannot be negative."
                         "'sampling_strategy' contains some negative value: {}"
                         .format(sampling_strategy))
    sampling_strategy_ = {}
    if sampling_type == 'over-sampling':
        n_samples_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        for class_sample, n_samples in sampling_strategy.items():
            if n_samples < target_stats[class_sample]:
                raise ValueError("With over-sampling methods, the number"
                                 " of samples in a class should be greater"
                                 " or equal to the original number of samples."
                                 " Originally, there is {} samples and {}"
                                 " samples are asked.".format(
                                     target_stats[class_sample], n_samples))
            if n_samples > n_samples_majority:
                warnings.warn("After over-sampling, the number of samples ({})"
                              " in class {} will be larger than the number of"
                              " samples in the majority class (class #{} ->"
                              " {})".format(n_samples, class_sample,
                                            class_majority,
                                            n_samples_majority))
            sampling_strategy_[class_sample] = (
                n_samples - target_stats[class_sample])
    elif sampling_type == 'under-sampling':
        for class_sample, n_samples in sampling_strategy.items():
            if n_samples > target_stats[class_sample]:
                raise ValueError("With under-sampling methods, the number of"
                                 " samples in a class should be less or equal"
                                 " to the original number of samples."
                                 " Originally, there is {} samples and {}"
                                 " samples are asked.".format(
                                     target_stats[class_sample], n_samples))
            sampling_strategy_[class_sample] = n_samples
    elif sampling_type == 'clean-sampling':
        # FIXME: Turn into an error in 0.6
        warnings.warn("'sampling_strategy' as a dict for cleaning methods is "
                      "deprecated and will raise an error in version 0.6. "
                      "Please give a list of the classes to be targeted by the"
                      " sampling.", DeprecationWarning)
        # clean-sampling can be more permissive since those samplers do not
        # use samples
        for class_sample, n_samples in sampling_strategy.items():
            sampling_strategy_[class_sample] = n_samp
        raise NotImplementedError

    return sampling_strategy_

    #mahalanobis function computations
   def mahalanob(self, xd, meanCol, IC):
        m = []
        for i in range(xd.shape[0]):
            m.append(mahalanobis(xd.iloc[i, :], meanCol, IC) ** 2)
        return (m)

    #generate new data by average same labels
   def gen_child_data(self, xdata, n_samples):
    n_samples_generated = 0
    cols = list(xdata.columns.values)
        #if(len(n_samples) < math.floor(len(xdata)/2)):
    newdat = xdata.groupby('mahrank').mean().reset_index()
    newdat = newdat[cols]
    uid = xdata.tail(1).iloc[0]['uid'] + 1
    newdat['uid'] = uid
    xdata['curl'] = 0
    newdat['curl'] = 1
    newdat['par1'] = 1
    newdat['par2'] = 2
    gen_data = newdat
    n_samples_generated += len(gen_data)
    xdata = pd.concat([xdata,gen_data])

    while(n_samples_generated < n_samples):
         #select last generated child(ren) data to be merged with its parents
         extdat = xdata.loc[xdata['curl'] == 1]

         #extract all children
         uids = extdat.uid.unique()
         uid = extdat.tail(1).iloc[0]['uid'] + 1
         #create empty gen array
         #gen_data = np.empty((0, X_class.shape[1]))
         xdata['curl'] = 0
         #gen_data['curl'] = 0
         for l in uids:
             subdat = extdat.loc[extdat['uid'] == l]
             #find the parents of the child
             par1 = subdat.tail(1).iloc[0]['par1']
             par2 = subdat.tail(1).iloc[0]['par2']
             mainpars = [par1,par2]
             #select parents and merge with current data and generate new data
             for gp in mainpars:
              subdat2= xdata.loc[xdata['uid'] == gp]
              mdat = pd.concat([subdat2, subdat])
              newdat = mdat.groupby('mahrank').mean().reset_index()
              newdat = newdat[cols]
              newdat['uid'] = uid
              uid = uid + 1
              #convert child data to old and make new data current
              #subdat['curl'] = 0
              newdat['curl'] = 1
              newdat['par1'] = subdat.tail(1).iloc[0]['uid']
              newdat['par2'] = subdat2.tail(1).iloc[0]['uid']
              #gen_data = np.append(gen_data, [newdat], axis=0)
              gen_data = pd.concat([gen_data, newdat])
              xdata = pd.concat([xdata, newdat])
         n_samples_generated += len(gen_data)

    #actual data to return
    acc_dat = xdata.loc[xdata['curl'] == 0]
    lastdat = xdata.loc[xdata['curl'] == 1]
    act = (len(lastdat)) - (len(gen_data) - n_samples)
    remids = lastdat.uid.unique()
    tk = math.ceil(act / len(remids))
    #select same number from last batch of syn data
    rem_data = pd.DataFrame()
        #np.zeros((tk*len(remids), acc_dat.shape[1]))
    for l in remids:
        subdat = lastdat.loc[lastdat['uid'] == l]
        remdat = subdat.iloc[0:tk]
        rem_data = rem_data.append(remdat)

    #final data
    acc_dat = acc_dat.append(rem_data)
    acc_dat =  acc_dat.drop(['mahrank','mahd','uid', 'par1', 'par2', 'curl'], axis=1)

    return acc_dat

   def fit_sample(self, X,y):
        """A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Points from which the points will be created.

        y_type : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.
        n_samples : int
            The number of samples to generate.


        Returns
        -------
        X_new : {ndarray, sparse matrix}, shape (n_samples_new, n_features)
            Synthetically generated samples.

        y_new : ndarray, shape (n_samples_new,)
            Target values for synthetic samples.

        """
        #SET INDEX AS counting numbers for future iterrows
        X = X.assign(id=range(X.shape[0]))
        X.set_index('id', inplace=True)

        sampling_type = 'over-sampling'
        X_resampled = X.copy()
        y_resampled = y.copy()

        sampling_strategy = self.sampling_strategy_majority(y, sampling_type)
        sampling_strategy_ = self.sampling_strategy_dict(sampling_strategy, y, sampling_type)

        for class_sample, n_samples in sampling_strategy_.items():
            # select minority class
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = safe_indexing(X, target_class_indices)



        # compute mahalanobis distance

        x = X_class  # .iloc[:, 1:]
        Sx = x.cov().values
        det = numpy.linalg.det(Sx)
        if det != 0:
         Sx = numpy.linalg.inv(Sx)
        else:
         # save column name
         coname = list(X_class.columns.values)
         X_class[coname] = X_class[coname].applymap(np.int64)
         # drop correlated features
         # Create correlation matrix
         corr_matrix = X_class.corr().abs()

         # Select upper triangle of correlation matrix
         upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

         # Find features with correlation greater than 0.95
         to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

         # Drop features
         X_class.drop(X_class.columns[to_drop], axis=1)

         x = X_class
         Sx = x.cov().values
         Sx = sp.linalg.inv(Sx)

        mean = x.mean().values

        md = self.mahalanob(x, mean, Sx)
        # add mahd values
        X_class = X_class.assign(mahd=md)

        # add the ranks of the mah dist val
        X_class['mahrank'] = X_class['mahd'].rank(ascending=False)
        # sort descending order
        X_class = X_class.sort_values(by=['mahd'], ascending=False)
        # compute median to use for partitioning
        medd = X_class.median()['mahrank']
        print(medd)
        # unique identifier
        X_class['uid'] = 1
        # assign labels to partition into two groups
        k = 1
        w = 1
        for i, row in X_class.iterrows():
            X_class.loc[i, 'mahrank'] = k
            # unique identifier
            X_class.loc[i, 'uid'] = w
            k = k + 1
            if (X_class.loc[i, 'mahrank'] == math.ceil(medd)):
                k = 1
                X_class.loc[i, 'mahrank'] = k
                # unique identifier
                w = 2
                X_class.loc[i, 'uid'] = w
                k = k + 1

        # add mom and dad columns no
        X_class['par1'] = 0
        X_class['par2'] = 0
        X_class['curl'] = 0

        #xdata = X_class

        X_new = self.gen_child_data(X_class, n_samples)

        y_new = np.array([class_sample] * np.sum(len(X_new)))

       # X_resampled = pd.concat([gen_data, newdat])
        X_resampled = np.vstack([X_resampled, X_new])
        y_resampled = np.hstack((y_resampled, y_new))

        # X_resampled = X_new
        # y_resampled = y_new

        return X_resampled, y_resampled
