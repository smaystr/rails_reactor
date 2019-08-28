class StandardScaler:
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
    
    def fit(self, data):
        if self.with_mean:
            self.mean = data.mean(axis=0)
        if self.with_std:
            self.std = data.std(axis=0)
            
    def transform(self, data):
        return (data - self.mean) / self.std

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

class LabelEncoder:
    def __init__(self):
        self.start = 0
    
    def fit(self, data, columns):
        if data.shape[1] == len(columns):
            self.unique = {i: {} for i in columns}
            for key, value in enumerate(columns):
                unique_columns = np.unique(data[:, key])
                for new_key, i in enumerate(unique_columns):
                    self.unique[value][i] = new_key + self.start
        else:
            raise Exception('wrong datashape or columns number')

    def transform(self, data, columns):
        if not set(columns).issubset(set(self.unique.key())):
            raise Exception('New column')
        for i in columns:
            for key, value in self.unique[i].items():
                data[data==key] = value
        return data.astype(float)
                        
    def fit_transform(self, data, columns):
        self.fit(data, columns)
        return self.transform(data, columns)
                            