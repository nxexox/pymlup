import logging
from typing import Union, Type

from mlup.web_app.api_collections import WebAppArchitecture
from mlup.web_app.architecture.worker_and_queue import WorkerAndQueueArchitecture
from mlup.web_app.architecture.batching import BatchingSingleProcessArchitecture
from mlup.web_app.architecture.directly_to_predict import DirectlyToPredictArchitecture

logger = logging.getLogger('MLup')


architectures = {
    WebAppArchitecture.directly_to_predict: DirectlyToPredictArchitecture,
    WebAppArchitecture.batching: BatchingSingleProcessArchitecture,
    WebAppArchitecture.worker_and_queue: WorkerAndQueueArchitecture,
}


def get_architecture(
    archi_type: Union[WebAppArchitecture, str]
) -> Union[
    Type[DirectlyToPredictArchitecture],
    Type[BatchingSingleProcessArchitecture],
    Type[WorkerAndQueueArchitecture]
]:
    if isinstance(archi_type, str):
        archi_type = WebAppArchitecture(archi_type)
    try:
        return architectures[archi_type]
    except KeyError:
        logger.error(
            f'Web App Architecture {archi_type} not found. '
            f'Use default Architecture {WebAppArchitecture.directly_to_predict}'
        )
        return architectures[WebAppArchitecture.directly_to_predict]
