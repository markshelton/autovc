#!/Anaconda3/env/honours python

"""model_comparer"""

#standard modules

#third party modules

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score,KFold

#local modules

class ModelComparer(object):

    def __init__(self, features, labels, models, scoring="roc_auc", plot=True, log=None):
        self.features = features
        self.labels = labels
        self.data = pd.concat([features, labels],axis=1)
        self.models = models
        self.scoring = scoring
        self.log = log
        self.results = self.compare_models(features, labels, models, scoring)
        if plot: self.plot_results(self.results)

    def compare_models(self,features, labels, models, scoring):
        self.log.info("model comparison starting")
        try:
            labels = labels.values.ravel()
            results = {}
            for name, model in models:
                self.log.info("model started\t| %s",name)
                try:
                    kfold = KFold(n=len(features))
                    cv_r = cross_val_score(model, features, labels,cv=kfold, scoring=scoring)
                    msg = "%s: %f (%f)" % (name,cv_r.mean(),cv_r.std())
                    results[name] = cv_r
                except:
                    self.log.error("model failed\t| %s",name,exc_info=1)
                    pass
                self.log.info("model successful\t| %s",msg)
        except:
            self.log.error("model comparison failed", exc_info=1)
        self.log.info("model comparison successful")
        return results

    def plot_results(self,results):
        self.log.info("plotting started")
        try:
            fig = plt.figure()
            fig.suptitle('Algorithm Comparison')
            ax = fig.add_subplot(111)
            plt.boxplot(list(results.values()))
            ax.set_xticklabels(results.keys())
            plt.show()
        except:
            self.log.error("plotting failed", exc_info=1)
            pass
        self.log.info("plotting successful")
