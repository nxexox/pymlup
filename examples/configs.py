import logging

from mlup.ml_model.model import MLupModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MLup')


if __name__ == "__main__":
    mlup_model = MLupModel(
        name='MyName',
        version='MyVersion',
        columns=[
            {'name': 'col1', 'type': 'float'},
            {'name': 'col2', 'type': 'int', 'required': False, 'default': 10},
        ],
        auto_detect_predict_params=True,
        path_to_binary='/mldata/models/binary_cls_model.pckl',
        use_thread_loop=False,
    )
    mlup_model.load()

    # Load configs from config file
    mlup_model = MLupModel()
    # or mlup_model.config.load_from_json
    mlup_model.config.load_from_yaml("/Users/t.deys/Projects/MLup/mlup-ws/conf/hello_world.yaml")
    mlup_model.load()

    # Save configs to file
    # or mlup_model.config.save_to_json
    mlup_model.config.save_to_yaml("/Users/t.deys/Projects/MLup/mlup-ws/conf/example_save_config.yaml")

    # You can load config and then change it
    mlup_model.config.load_from_yaml("/Users/t.deys/Projects/MLup/mlup-ws/conf/hello_world.yaml")
    mlup_model.name = 'NewName'
    mlup_model.load()
    mlup_model.config.save_to_yaml("/Users/t.deys/Projects/MLup/mlup-ws/conf/example_save_config.yaml")
