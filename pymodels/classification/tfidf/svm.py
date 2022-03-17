from sklearn import svm


class SVMModel:
    def __init__(self, config):
        self.config = config
        self.trained_model = None

    def train(self, train_x, train_y):
        assert self.trained_model is None
        sig = svm.SVC(decision_function_shape='ovr', max_iter=480,
                      class_weight="balanced", probability=True).fit(train_x, train_y)
        self.trained_model = sig

    def val(self, val_x, val_y):
        assert self.trained_model is not None
        preds = self.trained_model.predict_log_proba(val_x)
        return preds
