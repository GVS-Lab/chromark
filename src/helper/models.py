from copy import deepcopy

from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer


class ModelConfig(object):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_function: Module,
        inputs: Tensor = None,
        labels: Tensor = None,
        extra_features: Tensor = None,
        trainable: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.inputs = inputs
        self.labels = labels
        self.extra_features = extra_features
        self.trainable = trainable

        self.initial_weights = deepcopy(self.model.state_dict())

    def reset_model(self):
        self.model.load_state_dict(self.initial_weights)


class ExpConfig(object):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_function: Module,
        data_loader_dict: dict,
        data_key: str,
        label_key: str,
        index_key: str = None,
        train_model: bool = True,
        extra_feature_key: str = None,
    ):
        self.model_config = ModelConfig(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            trainable=train_model,
        )
        self.data_loader_dict = data_loader_dict
        self.data_key = data_key
        self.label_key = label_key
        self.extra_feature_key = extra_feature_key
        self.index_key = index_key
