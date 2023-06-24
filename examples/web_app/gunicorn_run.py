import logging

import uvicorn

from mlup.ml_model.model import MLupModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MLup')

mlup_model = MLupModel()
mlup_model.config.load_from_yaml("/Users/t.deys/Projects/MLup/mlup-ws/conf/hello_world.yaml")
mlup_model.load()
# For create WebAPp for uvicorn, need call load(). load call automate, when you call mlup_model.run_web_app().
mlup_model.web_app.load()

web_app = mlup_model.web_app.web_app

# gunicorn gunicorn_run:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8009
if __name__ == '__main__':
    uvicorn.run("__main__:web_app", host='0.0.0.0', port=8009, workers=2)
