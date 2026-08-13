# CHANGELOG

## 0.4.0

### Changed

* Migrated the web layer to FastAPI `>=0.140.0,<0.141.0` and Pydantic v2 (`>=2.13.0,<3.0.0`).
  Public classes, CLI commands, and endpoints (`/health`, `/info`, `/predict`, `/docs`,
  `/openapi.json`) are unchanged.
* Minimum supported Python version is now 3.10. Python 3.8/3.9 support and Pydantic v1
  compatibility remain on the pymlup 0.3.x line.
* Replaced the implicit `fastapi[all]` dependency with explicit `fastapi`, `pydantic`,
  `uvicorn`, and `starlette>=0.48.0` (`starlette` is pinned only because
  `status.HTTP_422_UNPROCESSABLE_CONTENT` doesn't exist before 0.48.0; `fastapi` alone would
  allow an older, incompatible `starlette`). The test suite now depends on `httpx` directly,
  since it's no longer pulled in transitively through `fastapi[all]`.
* Validation error responses now follow Pydantic v2's error taxonomy. The response shape
  (`{"detail": [...], "predict_id": ...}`) is unchanged, but `type`/`msg` values differ, e.g.
  `"value_error.missing"` -> `"missing"`, `"type_error.bool"` -> `"bool_parsing"`.
* **Behavior change:** predict columns/arguments marked as *not required* only ever got
  `default=None` in the generated Pydantic model, never an `Optional[...]` type. Pydantic v1
  inferred implicit optionality from a `None` default and accepted an explicit JSON `null` for
  such a field; Pydantic v2 does not infer this anymore, so sending an explicit `null` for a
  not-required column now returns a 422 (omitting the key entirely still works as before).
* **Behavior change:** a custom `str`-typed column no longer silently coerces a non-string
  input (e.g. a JSON number) into a string; Pydantic v2 rejects it with a `string_type` error
  instead of the implicit `str(value)` coercion Pydantic v1 performed.
* The OpenAPI schema for optional fields (e.g. `predict_id` in the error response) now uses
  Pydantic v2's `anyOf: [{"type": ...}, {"type": "null"}]` nullable representation instead of a
  plain typed field.
* `status.HTTP_422_UNPROCESSABLE_ENTITY` replaced with `status.HTTP_422_UNPROCESSABLE_CONTENT`
  in error responses (same `422` status code; the old name is deprecated in current Starlette).

### Fixed

* `mlup/web/api_docs.py` raised `KeyError: None` when generating the OpenAPI schema for a
  predict parameter that has neither a type annotation nor a default value mapping to a
  supported primitive type (e.g. `def predict(self, X, tag=None)`). Such fields now get a
  schema without an explicit `type` (valid JSON Schema for "any type") instead of crashing.
* Deprecated Pydantic v1 serialization calls (`.dict()`) in `mlup/web/api_errors.py` and
  `mlup/web/app.py` replaced with `.model_dump()`.
* The `run_easy.ipynb`/`run_worker_and_queue.ipynb`/`run_batching.ipynb` integration test
  notebooks all hardcoded the same port (`8009`) also used by dozens of unrelated unit tests,
  and never stopped their web server if the predict request failed - an earlier failure could
  leave an orphaned server squatting the port for later tests. Each notebook now uses its own
  dedicated port, polls `/health` before calling `/predict`, and always calls
  `up.stop_web_app()` in a `finally` block.
* `tests/unit_tests/console_scripts/test_run.py` started a real server process and then did a
  fixed `time.sleep(10)`/`time.sleep(20)` followed by an unguarded `requests.get(.../health)`
  (no timeout). Under a slow/loaded machine this could hang the test run indefinitely instead
  of failing. Replaced with a bounded `/health`-polling helper (same fix pattern as the
  notebook tests above), and its subprocess cleanup now joins with a timeout and escalates to
  `kill()` instead of a bare `terminate()` + fixed sleep, so the next test can't be blocked by
  a not-yet-released port.
* `tests/integration_tests/console_commands/test_run_cli_overrides.py`'s `_wait_for()` helper
  called `requests.get`/`.post` without a per-request timeout; a server that accepted the
  connection but never responded could hang the test run indefinitely despite the retry loop
  being bounded by attempt count. Added `timeout=5` to the request call.

### Removed

* Python 3.8 and 3.9 support (`requires-python` is now `>=3.10`).
* Pydantic v1 support. No `pydantic.v1` compatibility layer was added - dual v1/v2 support
  isn't viable on current FastAPI (confirmed experimentally; see
  `research/fastapi-pydantic-v2-spike.md`).

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