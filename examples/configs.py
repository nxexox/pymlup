import mlup
from mlup import constants


if __name__ == "__main__":
    up = mlup.UP(
        conf=mlup.Config(
            name='MyName',
            version='MyVersion',
            columns=[
                {'name': 'MinTemp', 'type': 'float', 'required': False, 'default': 1.4},
                {'name': 'MaxTemp', 'type': 'float'},
                {'name': 'Humidity9am', 'type': 'float'},
                {'name': 'Humidity3pm', 'type': 'float'},
                {'name': 'Pressure9am', 'type': 'float'},
                {'name': 'Pressure3pm', 'type': 'float'},
                {'name': 'Temp9am', 'type': 'float'},
                {'name': 'Temp3pm', 'type': 'float'},
            ],
            auto_detect_predict_params=True,
            storage_type=constants.StorageType.disk,
            storage_kwargs={
                'path_to_files': '../mldata/models/scikit-learn-binary_cls_model.pckl',
                'files_mask': 'scikit-learn-binary_cls_model.pckl',
            },
            use_thread_loop=False,
        )
    )
    up.ml.load()
    up.to_yaml('example-config.yaml')
    up.to_json('example-config.json')

    # Load configs from config file
    new_up = mlup.UP.load_from_yaml('example-config.yaml', load_model=True)

    # Save configs to file
    # or up.config_provider.save_to_yaml
    up.to_yaml('example-save-config.yaml')

    # You can load config and then change it
    up_for_change = up.load_from_yaml('example-config.yaml', load_model=True)
    up_for_change.name = 'NewName'
    up_for_change.to_yaml('example-changed-config.yaml')
