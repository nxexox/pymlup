# Binarizers

There are a huge number of ways to deliver a machine learning model to the server.
But in all cases, you need to deliver some data to this model: a binary pickle object, weights and layers of the neural network, text configuration of the trained model, and others.
These are always some files or one file.

When your machine learning model application runs, it loads the model into memory using this data.

There are many ways to turn a trained model into a file and back again.
Only some of them are supported out of the box in mlup:

* [pickle](https://github.com/nxexox/pymlup/blob/main/mlup/ml/binarization/pickle.py) - docs is [here](https://docs.python.org/3/library/pickle.html);
* [joblib](https://github.com/nxexox/pymlup/blob/main/mlup/ml/binarization/joblib.py) - docs is [here](https://joblib.readthedocs.io/en/latest/);
* [lightgbm](https://github.com/nxexox/pymlup/blob/main/mlup/ml/binarization/lightgbm.py) - docs is [here](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html);
* [torch all formats](https://github.com/nxexox/pymlup/blob/main/mlup/ml/binarization/torch.py) - docs is [here](https://pytorch.org/tutorials/beginner/saving_loading_models.html);
* [tensorflow all formats](https://github.com/nxexox/pymlup/blob/main/mlup/ml/binarization/tensorflow.py) - docs is [here](https://www.tensorflow.org/tutorials/keras/save_and_load);
* [onnx](https://github.com/nxexox/pymlup/blob/main/mlup/ml/binarization/onnx.py) - docs is [here](https://onnxruntime.ai/docs/get-started/with-python.html);

You can select the one you need in the configuration using `binarization_type`. In `binarization_type` you can specify:
* "auto" - default. In this case, mlup will try to automatically select a binarizer based on the name of the model file and its contents.
If it fails, a [ModelLoadError](https://github.com/nxexox/pymlup/blob/main/mlup/errors.py) exception will be thrown.
* select one of the mlup binarizers and specify it using [mlup.constants.BinarizationType](https://github.com/nxexox/pymlup/blob/main/mlup/constants.py).
* specify your own binarizer - the full python import line for your binarizer.

## Auto

This is the default way to select a binarizer for a model. It uses well-known mlup binarizers, each of which has some knowledge about its model storage format.

Based on this knowledge, the binarizer analyzes the model name and the contents of the model source files and returns a probability of the degree of confidence that it can load this model.
The binarizer with the greatest confidence and tries to load. And if all binarizers returned confidence 0, then the automatic selection is considered unsuccessful.

Not all binarizers participate in the automatic binarizer search, but only:
* [mlup.ml.binarization.pickle.PickleBinarizer](https://github.com/nxexox/pymlup/blob/main/mlup/ml/binarization/pickle.py)
* [mlup.ml.binarization.joblib.JoblibBinarizer](https://github.com/nxexox/pymlup/blob/main/mlup/ml/binarization/joblib.py)
* [mlup.ml.binarization.lightgbm.LightGBMBinarizer](https://github.com/nxexox/pymlup/blob/main/mlup/ml/binarization/lightgbm.py)
* [mlup.ml.binarization.torch.TorchBinarizer](https://github.com/nxexox/pymlup/blob/main/mlup/ml/binarization/torch.py)
* [mlup.ml.binarization.tensorflow.TensorFlowBinarizer](https://github.com/nxexox/pymlup/blob/main/mlup/ml/binarization/tensorflow.py)
* [mlup.ml.binarization.tensorflow.TensorFlowSavedBinarizer](https://github.com/nxexox/pymlup/blob/main/mlup/ml/binarization/tensorflow.py)
* [mlup.ml.binarization.onnx.InferenceSessionBinarize](https://github.com/nxexox/pymlup/blob/main/mlup/ml/binarization/onnx.py)

If the model is not already loaded into memory, binarizers do not load it into memory for analysis.
Instead, they can read up to 100 bytes from the beginning of the file and up to 100 bytes from the end of the file.
Therefore, the memory used by the application should not increase when you use automatic binarizer selection.

After successful selection, mlup will install the found binarizer in your application config in `binarization_type`.
If your application saves the config after this, then it will no longer contain “auto”, but the selected binarizer.
This is done so as not to select a binarizer for the same model several times.

**IMPORTANT: automatic selection does not guarantee 100% correct determination of the desired binarizer and may make mistakes. Please be careful and check in the logs and config which binarizer mlup chose for you.**

## Mlup binaraizers

### mlup.ml.binarization.pickle.PickleBinarizer

Pickle binarizer (`mlup.ml.binarization.pickle.PickleBinarizer`) is quite simple. All of his code actually converges on a call to `pickle.load()`.

#### Auto search

To check whether the binarizer can load a model from a file, the binarizer uses the file extension and the first and last bytes of the file.

According to [pickle documentation](https://peps.python.org/pep-3154/), the first and last bytes are always the same.
This sign adds 90% confidence.

The idea is to check the file for these bytes with similar code:
```python
import pickletools

is_pickle_file_confidence: float = 0.0

with open("file.pickle", "rb") as f:
    start = f.read(1)
    f.seek(-2, 2)
    end = f.read()
    file_bytes = start + end
    start_opcode = pickletools.code2op.get(file_bytes[0:1].decode('latin-1'))
    end_opcode = pickletools.code2op.get(file_bytes[-1:].decode('latin-1'))
    if start_opcode.name == 'PROTO' and end_opcode.name == 'STOP':
        is_pickle_file_confidence = 0.9
```

In addition to this, the file extension will be checked. If it is `.pckl` or `.pkl`, confidence increases by 5%.

```python
is_pickle_file_confidence: float

if 'file.pickle'.endswith('.pckl') or 'file.pickle'.endswith('.pkl'):
    is_pickle_file_confidence += 0.05
```

_Perhaps in the future we will conduct more in-depth research on binarization in this way and improve the analysis code._

---

### mlup.ml.binarization.joblib.JoblibBinarizer

Joblib binarizer is a copy of Pickle binarizer, except calling `joblib.load()` instead of `pickle.load()`.

#### Auto search

Although joblib is involved in automatic selection, it completely copies the pickle analysis method.

### mlup.ml.binarization.lightgbm.LightGBMBinarizer

The LightGBM binarizer builds a model based on the `lightgbm.Booster` constructor.

```python
import lightgbm as lgb

path: str
raw_model_data: str

if path:
    model = lgb.Booster(model_file=path)
model = lgb.Booster(model_str=raw_model_data)
```

#### Auto search

LightGBM has a text format for saving settings. mlup reads the first 100 bytes of the settings file and looks for a familiar signature there.
In the current implementation, this is a search for a string starting with "version" in the first 100 bytes of the file. But in the future this signature may change and become more complex.
This will definitely be written about in the documentation.
This sign brings 80% confidence.

Searching for a signature looks something like this:
```python
is_pickle_file_confidence: float = 0.0

with open('model_path.txt', 'r') as f:
    file_data = f.read(100)
*rows, other = file_data.split('\n', 5)
if any([r.split('=')[0] == 'version' for r in rows]):
    is_pickle_file_confidence = 0.8

```

In addition to this, the file extension will be checked. If it is `.txt`, confidence increases by 5%.

```python
is_pickle_file_confidence: float

if 'file.txt'.endswith('.txt'):
    is_pickle_file_confidence += 0.05
```

_Perhaps in the future we will conduct more in-depth research on binarization in this way and improve the analysis code._

---

### mlup.ml.binarization.torch.TorchBinarizer

Python has its own way of saving and loading a model. mlup uses exactly this. You can read more about ways to save and load your model in torch in the [documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

The entire binarizer code can be reduced to approximately the following lines:
```python
import torch

with open('file_path.pth', 'rb') as f:
    model = torch.load(f)
model.eval()
```

#### Auto search

This binarizer is involved in automatic selection. Based solely on our observations, torch models saved in the standard way have the same bytes at the beginning and end of the file.
Similar to Pickle binarization.

Considering that there is no official confirmation of this observation, this sign only adds 50% confidence.
Also, the `mlup.ml.binarization.tensorflow.TensorFlowBinarizer` binarizer is based on the same feature and the same bytes.

The following code does this check:
```python
is_pickle_file_confidence: float = 0.0

with open('model.pth', 'rb') as f:
    first_bytes = f.read(5)

if first_bytes.raw_data[:3] == b'PK\x03':
    is_pickle_file_confidence = 0.5
```

In addition to this, the file extension will be checked. If it is `.pth`, confidence increases by 30%.

```python
is_pickle_file_confidence: float

if 'file.pth'.endswith('.pth'):
    is_pickle_file_confidence += 0.3
```

_Perhaps in the future we will conduct more in-depth research on binarization in this way and improve the analysis code._

---

### mlup.ml.binarization.torch.TorchJITBinarizer

torch has several ways to save a model. JIT format is one of them. (You can read more in [torch documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html)).

In fact, the code for loading the JIT model differs from the code for loading the `.pth` torch model only in calling the required function from the torch framework.

```python
import torch

with open('file_path_jit.pth', 'rb') as f:
    model = torch.jit.load(f)
model.eval()
```

#### Auto search

This binarizer is not involved in the automatic selection of a binarizer, because we were unable to find distinctive features of this format.

_Perhaps in the future we will conduct more in-depth research on binarization in this way and add it._

---

### mlup.ml.binarization.tensorflow.TensorFlowBinarizer

tensorflow has its own way of saving and loading the model. mlup uses exactly this. You can read more about ways to save and load your model in torch in [documentation](https://www.tensorflow.org/tutorials/keras/save_and_load).

The entire binarizer code can be reduced to approximately the following lines:
```python
import tensorflow

model = tensorflow.keras.models.load_model('model.keras')
``` 

#### Auto search

This binarizer is involved in automatic selection. Based solely on our observations, tensorflow models saved in the standard way have the same bytes at the beginning and end of the file.
Similar to Pickle binarization.

Considering that there is no official confirmation of this observation, this sign only adds 50% confidence.
Also, the `mlup.ml.binarization.torch.TorchBinarizer` binarizer is based on the same feature and the same bytes.

The following code does this check:
```python
is_pickle_file_confidence: float = 0.0

with open('model.pth', 'rb') as f:
    first_bytes = f.read(5)

if first_bytes.raw_data[:3] == b'PK\x03':
    is_pickle_file_confidence = 0.5
```

In addition to this, the file extension will be checked. If it is `.keras` or `.h5`, confidence increases by 30%.

```python
is_pickle_file_confidence: float

if 'file.h5'.endswith('.keras') or 'file.h5'.endswith('.h5'):
    is_pickle_file_confidence += 0.3
```

_Perhaps in the future we will conduct more in-depth research on binarization in this way and improve the analysis code._

---

### mlup.ml.binarization.tensorflow.TensorFlowSavedBinarizer

tensorflow has several ways to save a model. Saved format is one of them. (You can read more in [torch documentation](https://www.tensorflow.org/tutorials/keras/save_and_load)).

In fact, the code for loading the saved model differs from the code for loading the `.keras` tensorflow model only in calling the required function from the tensorflow framework.

```python
import tensorflow

model = tensorflow.saved_model.load('model.pb')
```

#### Auto search

In the automatic selection of a binarizer, this binarizer is based only on the `.pb` file extension.
This sign adds only 30% confidence.

```python
is_pickle_file_confidence: float = 0.0

if 'file.pth'.endswith('.pb'):
    is_pickle_file_confidence = 0.3
```

_Perhaps in the future we will conduct more in-depth research on binarization in this way and improve the analysis code._

---

### mlup.ml.binarization.onnx.InferenceSessionBinarizer

Onnx is one of the most popular formats for saving models.
It can be used to save models from different frameworks, such as using pickle.

But the Python implementation has a different interface for using the loaded model than a simple call to the predict method with data passing.
In the standard case, the onnx model predictor looks like this:
```python
import onnxruntime

model = onnxruntime.InferenceSession('model.onnx')

input_name = model.get_inputs()[0].name
pred = model.run(None, {input_name: [[1, 2, 3], [4, 5, 6]]})
```

Additional actions required: receiving inputs. To reduce the predictor to 1 action, mlup has its own wrapper for the onnx model:
```python
import onnxruntime


class _InferenceSessionWithPredict(onnxruntime.InferenceSession):
    def predict(self, input_data):
        input_name = self.get_inputs()[0].name
        # Return model predict response in first item and all classes in second item
        res = self.run(None, {input_name: input_data})
        if len(res) > 1:
            return res[0]
        return res
```

The current mlup interface does not allow adding support for multiple inputs to the onnx model. This is due to the difference in incoming data.
For the onnx model, for each input it is necessary to transmit data isolated for this input on all objects.
And mlup takes all the data of one object together, and passes it to the model together.

For example:
```python
onnx_inputs = ['input1', 'input2', 'input3']
obj1 = [1, 2, 3]
obj2 = [4, 5, 6]
# For onnx need data format
for_onnx_model = [{n: list(features)} for n, features in zip(onnx_inputs, zip(obj1, obj2))]

# For mlup standart model
for_mlup_model = [obj1, obj2]
print(for_onnx_model)
print(for_mlup_model)
```

When working with a Python List, turning `for_mlup_model` into `for_onnx_model` is easy.
But when the data arrives at the model, it has already been processed by the data transformer (See [Data Transformers](data_transformers.md)).
Including a custom data transformer. Therefore, you cannot add generic code here to convert `for_mlup_model` to `for_onnx_model`.
If you add code based on known mlup data transformers, then onnx models become limited to using these data transformers.

**Be careful! If you have multiple inputs, you can add these transformations to your first neural network layer.**

The final code for loading onnx models is similar to this:
```python
import onnxruntime


class _InferenceSessionWithPredict(onnxruntime.InferenceSession):
    def predict(self, input_data):
        input_name = self.get_inputs()[0].name
        # Return model predict response in first item and all classes in second item
        res = self.run(None, {input_name: input_data})
        if len(res) > 1:
            return res[0]
        return res

model = _InferenceSessionWithPredict("/path/to/my/model.onnx")
```

#### Auto search

This binarizer is involved in automatic selection. onnx has its own way of checking the correctness of the onnx file, which is what the binarizer uses.
This sign brings 90% confidence.

```python
import onnx

is_pickle_file_confidence: float = 0.0
path = 'model.onnx'

try:
    onnx.checker.check_model(path)
    is_pickle_file_confidence = 0.9
except Exception:
    pass
```

In addition to this, the file extension will be checked. If it is `.onnx`, confidence increases by 5%.

```python
is_pickle_file_confidence: float

if 'model.onnx'.endswith('.onnx'):
    is_pickle_file_confidence += 0.05
```

_Perhaps in the future we will conduct more in-depth research on binarization in this way and improve the analysis code._

---

## Custom binarizer

If the capabilities of mlup binarizers are not enough for you, you can write your own binarizer.

The binarizer interface is very simple:
```python
# my_module.py
from typing import Any
from mlup.constants import LoadedFile
from mlup.ml.binarization.base import BaseBinarizer


class MyBinarizer(BaseBinarizer):
    @classmethod
    def deserialize(cls, data: LoadedFile) -> Any:
        pass
```

And specify the path to import your module in `binarization_type`: `my_module.MyBinarizer`.

**IMPORTANT: a binarizer written independently must be available for import on the server on which you run the mlup application.**

The easiest way to do this is to create your own python library with your binarizers and other useful classes and install it on your server along with the pymlup library.
