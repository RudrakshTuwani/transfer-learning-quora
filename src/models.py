from sklearn.linear_model import LogisticRegressionCV
from fastai.text import *


def get_report(y_true, y_pred, y_prob):
    acc = metrics.accuracy_score(y_true, y_pred)
    nll = metrics.log_loss(y_true, y_prob)

    print("Accuracy: ", acc)
    print("Negative Log loss: ", nll)
    print(metrics.classification_report(y_true, y_pred))
    
    return acc, nll


def nn_evaluate(dl, model):
    y_prob, y_true = list(), list()
    for x,y in dl:
        y_prob.append(to_np(model(to_gpu(Variable(x)))))
        y_true.append(to_np(y))

    y_prob, y_true = np.concatenate(y_prob), np.concatenate(y_true)
    y_pred = y_prob > 0.5
    
    get_report(y_true, y_pred, y_prob)


class ArCosModel(LogisticRegressionCV):
    def __init__(self, md, Cs=np.logspace(-5, 5, 20), cv=10, class_weight=None,
                 random_state=None, n_jobs=-1, verbose=1, scoring='neg_log_loss', **args):
        self.trn = md.trn_dl
        self.val = md.val_dl
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
        
        self.acc, self.nll = get_report(self.val_targets, self.val_preds, self.val_probs)
        
        
class FullyConnectedNet(nn.Module):
    def __init__(self, emb_sz, nhl, drops):
        super().__init__()
        
        nhl.insert(0, emb_sz*2)
        self.fcn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(nhl[i-1], nhl[i]),
                nn.ReLU(inplace=True),
                 nn.BatchNorm1d(nhl[i]),
                nn.Dropout(drops[i-1], inplace=True)
            )
            for i in range(1, len(nhl))])
        self.out = nn.Linear(nhl[-1], 1)
        
    def forward(self, input):
        bs = input.size()[0]
        x = input.view(bs, -1)
        for fc in self.fcn: 
            x = fc(x)

        out = F.sigmoid(self.out(x)).view(-1)
        return out


class EmbedQuestionNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1,2,0), (1,)).view(bs,-1)
    
    def forward_once(self, inp):
        raw_outputs, outputs = self.encoder(inp)
        output = outputs[-1]
        sl, bs, _ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[-1], mxpool, avgpool], 1)
        
        return x
    
    def transform(self, dl):
        q1, q2, targs = [], [], []

        for i, ((x1, x2), y) in enumerate(dl):
            q1.append(to_np(self.forward_once(Variable(x1.transpose(0, 1)))))
            q2.append(to_np(self.forward_once(Variable(x2.transpose(0, 1)))))
            targs.append(y)

            if i % 1000 == 0:
                print(f"Completed {i+1}/{len(dl)} batches.")
        print(f"Completed all batches")

        qnemb = np.concatenate([np.concatenate(q1)[None], np.concatenate(q2)[None]]).transpose(1, 0, 2)
        targs = np.concatenate(targs)

        return qnemb, targs