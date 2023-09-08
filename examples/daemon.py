import mlup
from mlup import constants


up = mlup.UP(
    conf=mlup.Config(
        # Set your config, for work model and get model.
        storage_type=constants.StorageType.disk,
        storage_kwargs={
            'path_to_files': '../mldata/models/scikit-learn-binary_cls_model.pckl',
            'files_mask': 'scikit-learn-binary_cls_model.pckl',
        },
    )
)
up.ml.load()
up.web.load()

# If you want to run the application yourself, or add something else to it, use this variable.
# Example with uvicorn: uvicorn mlupapp:app --host 0.0.0.0 --port 80
app = up.web.app

if __name__ == '__main__':
    # Default port is 8009
    up.run_web_app(daemon=True)
