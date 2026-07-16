# CHANGELOG

## 0.3.0

* Dropped Python 3.7 support, added 3.12/3.13/3.14 support. The `tensorflow` extra is
  capped to `python_version<'3.14'` since TensorFlow doesn't publish 3.14 wheels yet.
* Bumped numpy, scikit-learn, tensorflow (tests extra) and onnxruntime floors/ceilings
  so the ML framework extras actually install on the newer Python versions.
* Fixed `TorchBinarizer`/`torch.load` breaking on torch>=2.6, which changed the
  `weights_only` default to `True`.
* Regenerated all test ML model fixtures via small standalone scripts in `mldata/`
  (`generate_*_model.py`), replacing the old Jupyter notebooks.
* Fixed a hang in `mlup make-app` integration tests caused by `subprocess.Popen(shell=True)`
  leaving an orphaned process after `os.kill`.
* Fixed `mlup.ml.model` using a bare `list[...]` annotation that broke on Python 3.8.

## 0.1.1

Mini change inner code

## 0.1.0

This is first version library