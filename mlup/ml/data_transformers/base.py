import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Union, List, Optional

from mlup.constants import ModelDataTransformerType


@dataclass
class BaseDataTransformer(metaclass=abc.ABCMeta):
    """Base class for transformers data for predict to model format and back."""
    dtype_name: Optional[str] = None
    dtype: Optional[Any] = None
    DATA_TYPE: ModelDataTransformerType = field(init=False)

    def __post_init__(self):
        if self.dtype_name and self.dtype is None:
            self.dtype = self.get_dtype()

    def get_dtype(self) -> Any:
        return None

    @abc.abstractmethod
    def transform_to_model_format(
        self,
        data: List[Union[Dict[str, Any], List[Any]]],
        columns: Optional[List[Dict[str, str]]] = None,
    ):
        pass

    @abc.abstractmethod
    def transform_to_json_format(self, data: Any):
        pass
