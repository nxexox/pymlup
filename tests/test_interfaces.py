import inspect
from typing import Callable

from mlup.constants import ModelLibraryType, ModelDataType
from mlup.interfaces import MLupModelInterface, MLupWebAppInterface
from mlup.ml_model.model import MLupModel
from mlup.web_app.app import MLupWebApp
from mlup.web_app.api_collections import ITEM_ID_COL_NAME, WebAppArchitecture


def assert_property(attr, interface, cls):
    sign_interface = inspect.signature(getattr(interface, attr).fget)
    sign_cls = inspect.signature(getattr(cls, attr).fget)
    assert sign_interface == sign_cls, f'Signatures {interface}.{attr}{sign_interface} ' \
                                       f'and {cls}.{attr}{sign_cls} do not match.'

    for p_type in ('fset', 'fdel'):
        try:
            sign_interface = inspect.signature(getattr(getattr(interface, attr), p_type))
            sign_cls = inspect.signature(getattr(getattr(cls, attr), p_type))
            assert sign_interface == sign_cls, f'Signatures [{p_type}] {interface}.{attr}{sign_interface} ' \
                                               f'and [{p_type}] {cls}.{attr}{sign_cls} do not match.'
        except TypeError:
            pass


def assert_attribute(attr, interface, cls, default):
    if default in {False, True, None}:
        assert getattr(interface, attr) is default, f'Expected attribute {interface}.{attr}={default}. ' \
                                                    f'But found {interface}.{attr}={getattr(interface, attr)}'
        assert getattr(interface, attr) is getattr(cls, attr), f'Expected attribute {cls}.{attr}={default}. ' \
                                                               f'But found {cls}.{attr}={getattr(cls, attr)}'
    else:
        assert getattr(interface, attr) == default, f'Expected attribute {interface}.{attr}={default}. ' \
                                                    f'But found {interface}.{attr}={getattr(interface, attr)}'
        assert getattr(interface, attr) == getattr(cls, attr), f'Expected attribute {cls}.{attr}={default}. ' \
                                                               f'But found {cls}.{attr}={getattr(cls, attr)}'


def assert_method(method_name, interface, cls):
    sign_interface = inspect.signature(getattr(interface, method_name))
    sign_cls = inspect.signature(getattr(cls, method_name))
    assert sign_interface == sign_cls, f'Signatures {interface}.{method_name}{sign_interface} ' \
                                       f'and {cls}.{method_name}{sign_cls} do not match.'


def test_mlup_model_interface_have_attributes():
    exclude_fields = {
        'web_app',
    }

    for attr_name, attr_value in MLupModel.__dict__.items():
        if attr_name in exclude_fields:
            continue
        if not attr_name.startswith('_'):
            if isinstance(attr_value, Callable):
                assert_method(attr_name, MLupModelInterface, MLupModel)
            elif isinstance(attr_value, property):
                assert_property(attr_name, MLupModelInterface, MLupModel)
            else:
                assert_attribute(attr_name, MLupModelInterface, MLupModel, attr_value)


def test_mlup_web_app_interface():
    exclude_fields = {
        'info', 'predict'
    }

    for attr_name, attr_value in MLupWebApp.__dict__.items():
        if attr_name in exclude_fields:
            continue
        if not attr_name.startswith('_'):
            if isinstance(attr_value, Callable):
                assert_method(attr_name, MLupWebAppInterface, MLupWebApp)
            elif isinstance(attr_value, property):
                assert_property(attr_name, MLupWebAppInterface, MLupWebApp)
            else:
                assert_attribute(attr_name, MLupWebAppInterface, MLupWebApp, attr_value)
