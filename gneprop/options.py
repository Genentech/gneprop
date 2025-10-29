from collections.abc import Sequence
from enum import Enum


class GNEPropTask(Enum):
    BINARY_CLASSIFICATION = 'binary_classification'
    REGRESSION = 'regression'
    MULTI_CLASSIFICATION = 'multi_classification'

    def get_metrics(self):
        if self.value in ('binary_classification', 'multi_classification'):
            return ['auc', 'ap', 'acc']
        elif self.value == 'regression':
            return ['mse', 'mae']

    def get_default_metric(self):
        if self.value in ('binary_classification', 'multi_classification'):
            return 'auc'
        elif self.value == 'regression':
            return 'mse'

    @staticmethod
    def validation_names(m):
        if isinstance(m, str):
            return 'val_' + m
        elif isinstance(m, Sequence):
            return ['val_' + i for i in m]

    @staticmethod
    def test_names(m):
        if isinstance(m, str):
            return 'test_' + m
        elif isinstance(m, Sequence):
            return ['test_' + i for i in m]

