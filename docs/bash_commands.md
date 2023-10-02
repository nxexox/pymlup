# Description of the bash commands

## mlup validate-config

This command allows you to validate your config file for correctness.

```bash
$ mlup validate-config /path/to/your/config/file.yaml
Load config from /path/to/your/config/file.yaml
Config is valid
$
$ mlup validate-config --type json /path/to/your/config/file.json
Load config from /path/to/your/config/file.json
Config is valid
```

So far, this command cannot check the validity of the values in the configuration file. The command simply checks that mlup can read and load this config.

This is due to the fact that not everything can be validated without running the code to load the model. But I'm working on creating full validation for each field, its valid values, and the cumulative values throughout the entire config.

If the config is not valid, this command will exit with exit code 1.

## mlup run

Perhaps the most interesting command, which allows you to run the application directly according to the model without a config.

```bash
$ mlup run -m /path/to/my/model.onnx
```

But this team is much broader. It can run the application:
* by model - `mlup run -m /path/to/my/model.onnx`;
* according to the config - `mlup run -c /path/to/my/config.yaml`;
* according to the pickle file with mlup.UP - `mlup run -b /path/to/my/binary.pickle`;

Also, you can replace any config value on the fly, directly in the arguments.
```bash
$ mlup run -c /path/to/my/config.yaml --up.port=8010 --up.use_thread_loop=False
```

The order of using the config in this case:
* The most priority config from the arguments `--up.<conf_name>`;
* Config from your configuration file;
* Default config value;

Examples for different data types:
```bash
$ mlup run -m /path/tomy/model.onnx \
  --up.port=8011 \
  --up.batch_worker_timeout=10.0 \
  --up.predict_method_name=\"__call__\" \
  --up.use_thread_loop=False \
  --up.columns='[{"name": "col", "type": "list"}]' \
  --up.uvicorn_kwargs='{"workers": 4, "timeout_graceful_shutdown": 10}'
```

You can specify `--verbose` to get full logging with debug loggers for debugging.
```bash
$ mlup run -m /path/to/my/model.onnx --verbose
```

## mlup make-app

If the `mlup run` command is not enough for you, you can create a `.py` file with your mlup application.
In which you can add any of your Python code. For example: add your own API handlers for a web application.

```bash
$ mlup make-app /path/to/your/app.py
App success created: /path/to/your/app.py
```

This command will generate the following code in the file `/path/to/your/app.py`:
```python
import mlup


# You can load the model yourself and pass it to the "ml_model" argument.
# up = mlup.UP(ml_model=my_model, conf=mlup.Config())
up = mlup.UP(
    conf=mlup.Config(
        # Set your config, for work model and get model.
        # You can use storage_type and storage_kwargs for auto_load model from storage.
    )
)
up.ml.load()
up.web.load()

# If you want to run the application yourself, or add something else to it, use this variable.
# Example with uvicorn: uvicorn test-app2:app --host 0.0.0.0 --port 80
app = up.web.app

if __name__ == '__main__':
    up.run_web_app()
```

You can control which application will be generated using arguments:
* `mlup make-app -ms /path/to/your/model.onnx /path/to/your/app.py` - will add model loading from disk to the application.
* `mlup make-app -сs /path/to/your/config.yaml /path/to/your/app.py` - will add config loading from disk to the application.
* `mlup make-app -bs /path/to/your/mlupUP.pickle /path/to/your/app.py` - will add loading mlup.UP from disk and unpickle to the application.

For example, the command `mlup make-app -ms /path/to/your/model.onnx /path/to/your/app.py` will create the file:
```python
import mlup
from mlup import constants


up = mlup.UP(
    conf=mlup.Config(
        # Set your config, for work model and get model.
        storage_type=constants.StorageType.disk,
        storage_kwargs={
            'path_to_files': '/path/to/your/model.onnx',
            'files_mask': 'model.onnx',
        },
    )
)
up.ml.load()
up.web.load()

# If you want to run the application yourself, or add something else to it, use this variable.
# Example with uvicorn: uvicorn test-app2:app --host 0.0.0.0 --port 80
app = up.web.app

if __name__ == '__main__':
    up.run_web_app()
```

If the file `/path/to/your/app.py` already exists, you can specify `--force` to overwrite it. Then the existing file will be deleted and a new one will be created.

Just like in `mlup run`, you can specify your own configuration options via the `--up.<conf_name>` argument.
```bash
$ mlup make-app -ms /path/to/your/model.onnx /path/to/your/app.py --up.port=8010 --up.use_thread_loop=False`
```

Will create the following file:
```python
import mlup
from mlup import constants


up = mlup.UP(
    conf=mlup.Config(
        # Set your config, for work model and get model.
        storage_type=constants.StorageType.disk,
        storage_kwargs={
            'path_to_files': '/path/to/your/model.onnx',
            'files_mask': 'model.onnx',
        },
        port=8010,
        use_thread_loop=False,
    )
)
up.ml.load()
up.web.load()

# If you want to run the application yourself, or add something else to it, use this variable.
# Example with uvicorn: uvicorn test-app2:app --host 0.0.0.0 --port 80
app = up.web.app

if __name__ == '__main__':
    up.run_web_app()
```

You can launch the application through the created file with one command:
```bash
$ python /path/to/your/app.py
```
