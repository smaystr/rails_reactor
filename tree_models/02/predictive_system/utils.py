from sklearn.preprocessing import LabelEncoder
import numpy as np


class SafeLabelEncoder(LabelEncoder):
        # found it somewhere and changed it slightly
    def get_unseen(self):
        return 99999

    def transform(self, y):

        classes = np.unique(y)

        unseen = self.get_unseen()
        if len(classes) >= unseen:
            raise ValueError('Too many factor levels in feature. Max is %i' % unseen)

        result = np.array([
            np.searchsorted(self.classes_, x) if x in self.classes_ else unseen
            for x in y
        ])

        return result


class SafeOneHot():
    def __init__(self, fill='Missing', unseen=99999):
        self.classes_ = np.array([])
        self.number_of_classes = 0
        self.fill = fill
        self.unseen = unseen

    def preprocess(self, y):
        y = np.array(y)
        y[y == np.nan] = self.fill
        return y

    def get_unseen(self):
        return self.unseen

    def fit(self, y):
        y = self.preprocess(y)
        self.classes_ = np.unique(y)
        self.number_of_classes = len(self.classes_)

    def transform(self, y):
        y = self.preprocess(y)
        transformed = np.zeros((y.shape[0], len(self.classes_)))

        for key, entry in enumerate(y):
            find_ind = np.argwhere(self.classes_ == entry)
            if len(find_ind) == 0:
                continue
            transformed[key][find_ind[0][0]] = 1
        return transformed

    def set_params(self, classes, fill):
        self.classes_ = classes
        self.fill = fill

    def get_params(self):
        return (self.classes_, self.fill)
