from enum import Enum


ITEM_ID_COL_NAME: str = 'mlup_item_id'


class WebAppArchitecture(Enum):
    directly_to_predict = 'directly_to_predict'
    worker_and_queue = 'worker_and_queue'
    batching = 'batching'
