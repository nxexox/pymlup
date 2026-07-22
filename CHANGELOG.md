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
* Fixed `mlup run`/`mlup make-app` `--up.<field>=value` CLI overrides always being passed to
  `mlup.Config` as raw strings: bool fields could never actually be set to `False` (e.g.
  `--up.use_thread_loop=False`), int/float fields crashed the app on load (e.g.
  `--up.max_thread_loop_workers=4`), and list/dict fields (`--up.columns`, `--up.uvicorn_kwargs`)
  silently kept the raw JSON string instead of the parsed value. Values are now coerced to each
  `Config` field's real type; conversion errors now produce a clean argparse error instead of a
  traceback.
* Fixed `examples/configs.py` setting a stray `up_for_change.name` attribute instead of
  `up_for_change.conf.name`, which meant `example-changed-config.yaml` never actually reflected
  the renamed model.
* Fixed several documentation inaccuracies: `file_mask` → `files_mask` in the storage_kwargs
  examples, the `/health` response key (`status_code` → `status`) in `docs/web_app_api.md`, the
  `/predict` handler name in `docs/python_interface.md`, a Cyrillic character in a
  `docs/bash_commands.md` CLI example, invalid (trailing-comma) JSON in the `/info` examples, and
  the documented `host`/`max_thread_loop_workers` defaults.

## 0.1.1

Mini change inner code

## 0.1.0

This is first version library