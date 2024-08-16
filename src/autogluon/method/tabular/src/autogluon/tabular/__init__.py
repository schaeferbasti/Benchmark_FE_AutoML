from src.autogluon.method.common.src.autogluon.common.features.feature_metadata import FeatureMetadata
from src.autogluon.method.common.src.autogluon.common.utils.log_utils import _add_stream_handler
from src.autogluon.method.core.src.autogluon.core.dataset import TabularDataset

try:
    from .version import __version__
except ImportError:
    pass

from .predictor import TabularPredictor

_add_stream_handler()
