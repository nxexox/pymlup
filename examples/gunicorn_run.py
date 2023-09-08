import mlup
from mlup import constants
import uvicorn


up = mlup.UP(
    conf=mlup.Config(
        # Set your config, for work model and get model.
        storage_type=constants.StorageType.disk,
        storage_kwargs={
            'path_to_files': 'mldata/models/scikit-learn-binary_cls_model.pckl',
            'files_mask': 'scikit-learn-binary_cls_model.pckl',
        },
    )
)
up.ml.load()
# For create WebAPp for uvicorn, need call load() - up.web.load().
# load call automate, when you call up.run_web_app().
up.web.load()

# If you want to run the application yourself, or add something else to it, use this variable.
# Example with uvicorn: uvicorn gunicorn_run:app --host 0.0.0.0 --port 80
app = up.web.app

# gunicorn gunicorn_run:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8009
if __name__ == '__main__':
    uvicorn.run("__main__:app", host='0.0.0.0', port=8009, workers=2)
