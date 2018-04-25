from sklearn.linear_model import LogisticRegressionCV
from fastai.text import *


class ArCosModel(LogisticRegressionCV):
    def __init__(self, trn, val, Cs=np.logspace(-5, 5, 20), cv=10, class_weight=None,
                 random_state=None, n_jobs=-1, verbose=1, scoring='neg_log_loss', **args):
        self.trn = trn
        self.val = val
        super().__init__(Cs=Cs, cv=cv, class_weight=class_weight, random_state=random_state, 
                         n_jobs=n_jobs, verbose=verbose, scoring=scoring, **args)
        
    def _cosine_similarity(self, dl):
        """ Gets cosine similarity between the vectors of question pairs. """

        similarities = list()
        targets = list()
        for i, (x, y) in enumerate(dl):
            cossim = F.cosine_similarity(x[:, 0, :], x[:, 1, :])
            similarities.append(to_np(cossim))
            targets.append(to_np(y))
            
            if i % 2000 == 0: print(f"Completed {i+1} batches")
        print("Completed all batches!")
        
        targets = np.concatenate(targets)
        similarities = np.concatenate(similarities).clip(min=-1, max=1)
        
        return similarities, targets
    
    def _angular_distance_cos(self, dl):
        """ Gets angular distances using cosine similarity. """
        
        # Get cosine similarities
        similarities, targets = self._cosine_similarity(dl)
        
        # Convert to angular distance
        angular_distances = np.arccos(similarities) / np.pi
        
        return similarities, targets, angular_distances

    def fit(self):
        """ Calculates angular distances and rescales them to probabilities using a
        logistic regression classifier. """
        
        trn_similarities, trn_targets, trn_angular_distances =\
            self._angular_distance_cos(self.trn)

        # Learn a linear function mapping distances to probabilities.
        super().fit(trn_angular_distances.reshape(-1, 1), trn_targets)
        
    def evaluate(self):
        val_similarities, val_targets, val_angular_distances =\
            self._angular_distance_cos(self.val)
        
        # Get in desired shape
        val_angular_distances = val_angular_distances.reshape(-1, 1) 
        
        # Predict probabilities and labels.
        self.val_probs = super().predict_proba(val_angular_distances)
        self.val_preds = super().predict(val_angular_distances)
        self.val_targets = val_targets
        
        # Metrics
        self.acc = metrics.accuracy_score(self.val_targets, self.val_preds)
        self.nll = metrics.log_loss(self.val_targets, self.val_probs)

        print("Accuracy: ", self.acc)
        print("Negative Log loss: ", self.nll)
        print(metrics.classification_report(self.val_targets, self.val_preds))