# CHANGELOG

## 0.3.1

### Fixed

* Reproducible `scikit-learn` Quick Start in `README.md`/`docs/quickstart.md` that doesn't
  implicitly depend on `requests` being installed. The previous example crashed with
  `ModuleNotFoundError: No module named 'requests'` right after the server started, on a
  bare `pip install pymlup`; the new one uses `curl` for `/health` and `/predict`.
* The `Documentation` project URL shown on PyPI (`https://github.com/nxexox/pymlup/docs`,
  a 404) now points at `https://mlup.org/`.
* The README logo used a relative image path, which rendered as a broken image on the PyPI
  project page (GitHub resolves it, PyPI doesn't); it's now an absolute URL.
* `docs/quickstart.md` inaccurately stated that mlup "tries the pickle binarizer" by
  default; corrected to describe the actual `binarization_type="auto"` auto-detection
  behavior, and a dead link to a non-existent binarizer file path was fixed alongside it.
* Assorted first-page documentation errors: grammar in the opening paragraphs of
  `docs/README.md`.

### Added

* PyPI `keywords`: `machine-learning`, `model-serving`, `inference`, `fastapi`, `rest-api`,
  `mlops`, `scikit-learn`, `pytorch`, `tensorflow`, `onnx`, `lightgbm`.
* `Topic :: Scientific/Engineering :: Artificial Intelligence` classifier.
* Additional `project.urls` entries: `Source`, `Issues`, `Changelog`.
* `site_description` for the MkDocs site.

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