import numpy as np


class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features=X.shape
        self._classes=np.unique(y)
        n_classes=len(self._classes)

        #init
        self._mean=np.zeros((n_classes,n_features),dtype=np.float64)
        self._var= np.zeros((n_classes,n_features), dtype=np.float64)
        self._priori=np.zeros(n_classes,dtype=np.float64)

        for c in self._classes:
            x_c = X[c==y]
            self._mean[c,:]=x_c.mean(axis=0)
            self._var[c,:] = x_c.var(axis=0)
            self._priori[c]= x_c.shape[0]/float(n_samples)


    def predict(self,X):
        y_pred= [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        posteriors = []
        for idx,c in enumerate(self._classes):
            prior = np.log(self._priori[idx])
            class_conditional = np.sum(np.log(self._pdf(idx,x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]


    def _pdf(self,class_idx, x):
        mean=self._mean[class_idx]
        var=self._var[class_idx]
        numerator= np.exp(-(x-mean)**2 / (2*var))
        denominator= np.sqrt(2*np.pi*var)
        return numerator / denominator

    def accuracy(self, y_true, y_predection):
        accuracy=np.sum(y_true==y_predection)/len(y_true)
        return accuracy

