import logging
from time import sleep

from mlup.ml_model.model import MLupModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MLup')


if __name__ == "__main__":
    mlup_model = MLupModel()
    mlup_model.config.load_from_yaml("/Users/t.deys/Projects/MLup/mlup-ws/conf/hello_world.yaml")
    mlup_model.load()
    mlup_model.run_web_app(daemon=True)

    sleep(3)
    print('Web application running is separate thread.')

    mlup_model.stop_web_app()
