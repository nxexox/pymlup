import logging

from mlup.ml_model.model import MLupModel
from mlup.web_app.api_collections import WebAppArchitecture

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MLup>')


if __name__ == "__main__":
    mlup_model = MLupModel()
    mlup_model.config.load_from_yaml("/Users/t.deys/Projects/MLup/mlup-ws/conf/hello_world.yaml")
    mlup_model.web_app.mode = WebAppArchitecture.batching
    mlup_model.load()
    mlup_model.run_web_app()
