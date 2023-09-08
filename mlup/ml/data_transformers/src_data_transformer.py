from typing import Any, List, Dict, Union, Optional

from mlup.constants import ModelDataTransformerType
from mlup.ml.data_transformers.base import BaseDataTransformer


class SrcDataTransformer(BaseDataTransformer):
    """Class for don't transform data for predict. Use sending client and returning model data."""
    DATA_TYPE: ModelDataTransformerType = ModelDataTransformerType.SRC_TYPES

    def transform_to_model_format(
        self,
        data: List[Union[Dict[str, Any], List[Any]]],
        columns: Optional[List[Dict[str, str]]] = None,
    ) -> List[Union[Dict[str, Any], List[Any]]]:
        return data

    def transform_to_json_format(self, data: Any) -> Any:
        return data
