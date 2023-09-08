import logging.config
from dataclasses import field, dataclass, InitVar
from pathlib import Path
from typing import Dict, Any, Optional, Union

from mlup.config import ConfigProvider, LOGGING_CONFIG
from mlup.ml.empty import EmptyModel
from mlup.ml.model import MLupModel, ModelConfig
from mlup.utils.loop import run_async
from mlup.web.app import MLupWebApp, WebAppConfig


logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('mlup')


def generate_default_config(path_to_file: Optional[str] = None) -> Optional[Dict]:
    up = UP(ml_model=EmptyModel())
    if not path_to_file:
        return up.to_dict()
    if path_to_file.endswith('json'):
        up.to_json(path_to_file)
    else:
        up.to_yaml(path_to_file)


@dataclass(kw_only=True)
class Config(ModelConfig, WebAppConfig):
    pass


@dataclass(kw_only=True, repr=False)
class UP:
    """This is main UP class.
    Create object UP with your ML model, set your settings and run your web app.

    """
    ml_model: InitVar[Any] = None
    conf: Config = None

    # Inner model data
    config_provider: ConfigProvider = field(init=False, default=None, repr=False)
    ml: MLupModel = field(init=False, default=None, repr=True)
    _wep_app: MLupWebApp = field(init=False, default=None, repr=True)

    def __post_init__(self, ml_model: Any):
        """Running custom construct code after call dataclass construct code"""
        if self.config_provider is None:
            self.config_provider = ConfigProvider(
                obj_for_config=self,
                configs_objects_map={"ml": ModelConfig, "web": WebAppConfig}
            )
        if self.conf is None:
            self.conf = Config()

        if isinstance(ml_model, MLupModel):
            self.ml = ml_model
            self.ml.conf = self.conf
        else:
            self.ml = MLupModel(
                ml_model=ml_model,
                conf=self.conf,
            )

    @property
    def web(self) -> MLupWebApp:
        """MLupWebApp object"""
        if self._wep_app is None:
            self._wep_app = MLupWebApp(ml=self.ml, conf=self.conf)
        return self._wep_app

    @classmethod
    def load_from_dict(cls, conf: Dict, load_model: bool = True):
        """
        Load UP object from dict.

        :param Dict conf: Conf dict for loading UP object.
        :param bool load_model: Need load model to memory from storage? Default is True.

        """
        up = cls(ml_model=EmptyModel())
        up.config_provider.load_from_dict(conf)
        if load_model:
            up.ml.load()
        return up

    def to_dict(self):
        """UP config to dict"""
        return self.config_provider.get_config_dict()

    @classmethod
    def load_from_json(cls, conf_path: Union[str, Path], load_model: bool = True):
        """
        Load UP object from json config.

        :param Union[str, Path] conf_path: Path to JSON file with config.
        :param bool load_model: Need load model to memory from storage? Default is True.

        """
        up = cls(ml_model=EmptyModel())
        up.config_provider.load_from_json(conf_path)
        if load_model:
            up.ml.load()
        return up

    def to_json(self, path_to_file: Union[str, Path]):
        """
        Save up config to JSON file.

        :param Union[str, Path] path_to_file: Path to result config file.

        """
        return self.config_provider.save_to_json(path_to_file)

    @classmethod
    def load_from_yaml(cls, conf_path: Union[str, Path], load_model: bool = True):
        """
        Load UP object from yaml config.

        :param Union[str, Path] conf_path: Path to YAML file with config.
        :param bool load_model: Need load model to memory from storage? Default is True.

        """
        up = cls(ml_model=EmptyModel())
        up.config_provider.load_from_yaml(conf_path)
        if load_model:
            up.ml.load()
        return up

    def to_yaml(self, path_to_file: Union[str, Path]):
        """
        Save up config to YAML file.

        :param Union[str, Path] path_to_file: Path to result config file.

        """
        return self.config_provider.save_to_yaml(path_to_file)

    def predict(self, **predict_data):
        """
        Call model predict.

        :param predict_data: Data for predict. This data have same format as predict from web app.
        Example:
            up.predict(X=[[1, 2, 3], [4, 5, 6]])

        """
        return run_async(self.ml.predict, **predict_data)

    def predict_from(self, **predict_data):
        """
        Call model predict without transform data before model predict.

        :param predict_data: Data for predict with ML format, use numpy, pandas and etc libraries.
        Example:
            up.predict(X=numpy.array([[1, 2, 3], [4, 5, 6]]))

        """
        return run_async(self.ml.predict_from, **predict_data)

    async def async_predict(self, **predict_data):
        """
        Call model predict without use mlup.utils.loop.run_async.

        :param predict_data: Data for predict. This data have same format as predict from web app.
        Example:
            await up.predict(X=[[1, 2, 3], [4, 5, 6]])

        """
        return await self.ml.predict(**predict_data)

    def run_web_app(self, daemon: bool = False, force_load: bool = False):
        """
        Run web app with ML model.

        :param bool daemon: Run web app with daemon mode and unblock terminal. Default is False.
        :param bool force_load: Reload web_app before start, even web_app was loaded.

        """
        if not self.web.loaded or force_load:
            self.web.load()
        self.web.run(daemon=daemon)

    def stop_web_app(self, shutdown_timeout: Optional[float] = None):
        """Stop web app, if web app was runned in daemon mode."""
        self.web.stop(shutdown_timeout=shutdown_timeout)
