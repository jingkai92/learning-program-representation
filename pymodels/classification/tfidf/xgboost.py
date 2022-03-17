import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score


class XGBoostModel:
    def __init__(self, config):
        self.config = config
        self.trained_model = None
        # Multi Class Mode
        self.param = {
            'max_depth': self.config.tree_depth,  # the maximum depth of each tree
            'eta': 0.3,  # the training step for each iteration
            'silent': 1,  # logging mode - quiet
            'objective': self.config.xgboost_objective,  # error evaluation for multiclass training
            'num_class': self.config.class_num
        }  # the number of classes that exist in this datset
        # if self.config.class_num > 2:
        #     self.param['num_class'] = self.config.class_num

        self.num_round = self.config.round_num  # the number of training iterations

    def train(self, train_x, train_y):
        assert self.trained_model is None
        dtrain = xgb.DMatrix(train_x, label=train_y)
        self.trained_model = xgb.train(self.param, dtrain, self.num_round)

    def val(self, val_x, val_y):
        assert self.trained_model is not None
        dval = xgb.DMatrix(val_x, label=val_y)
        preds = self.trained_model.predict(dval)
        return preds
