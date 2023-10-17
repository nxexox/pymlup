# Storages

[Description of the application life cycle](life_cycle.md) describes the role of storage throughout mlup.
This component is needed to deliver the model from your storage to the local disk and transfer information about the downloaded file to the binarizer (See [Binarizers](binarizers.md)).
It can also load the contents of files into memory.

This can be any storage that your code can access from the server using any protocol.

At the moment there are no implementations for working with remote storages in mlup 🙃

So far, mlup supports two types of storage out of the box:

* [mlup.ml.storage.memory.MemoryStorage](https://github.com/nxexox/pymlup/blob/main/mlup/ml/storage/memory.py)
* [mlup.ml.storage.local_disk.DiskStorage](https://github.com/nxexox/pymlup/blob/main/mlup/ml/storage/local_disk.py)

You can select the desired storage using the `storage_type` configuration parameter.

The storage class may need additional parameters that govern its operation. For example, login and password for storage.
To pass these parameters to the storage class, you need to use the `storage_kwargs: Dict[str, Any]` configuration parameter.
Parameters from `storage_kwargs` are passed to the storage class constructor: `storage_class(**storage_kwargs)`.

## mlup.ml.storage.memory.MemoryStorage

This type is used when your model is already loaded into memory and does not need to be loaded and binarized.
It is the default in the mlup configuration.

For example:
```python
import mlup
from mlup.ml.empty import EmptyModel


up = mlup.UP(ml_model=EmptyModel())
```

The example above uses `storage_type=mlup.ml.storage.memory.MemoryStorage`.

This storage has no parameters for `storage_kwargs`.

## mlup.ml.storage.local_disk.DiskStorage

When you run a mlup application on the server, at that moment your model is not loaded into memory and needs to be loaded.

`mlup.ml.storage.local_disk.DiskStorage` finds the model file by path and mask on the local disk, and returns information on the found file, which mlup passes to the binarizer.

This storage has parameters for `storage_kwargs`:

* `path_to_files: str` - Required. The path to the folder with the model file or to the model file itself on disk.
* `file_mask: str` = Optional. By default `(\w.-_)*.pckl` is pickle. A regular expression that will be used to search for a file in `path_to_files`.
* `need_load_file: bool` - Optional. Default is False. If False, the binary model data itself will not be loaded into memory and the binarizer will load it itself. If True, then the storage will load the raw data into memory and give it to the binarizer.

**IMPORTANT!**

* By default, `mlup.ml.storage.local_disk.DiskStorage` does not load model data into memory. The binarizer loads them independently.
In this case, if your model weighs 1 GB, then you need at least 1 GB of RAM to run the application.
* If you specify `need_load_file=True`, and the model is loaded into memory by storage, memory duplication will occur: raw data loaded by storage and the model serialized from this data.
It turns out that the increase in memory consumed by the application for launching doubles the weight of the model.
If your model weighs 1 GB, then you will need at least 2 GB to run the application.
But after running, the memory consumption should drop back to 1 GB because mlup explicitly deletes the raw data after serialization and calls the garbage collector.

```python
import gc
from mlup.constants import LoadedFile

loaded_files: list[LoadedFile] = storage.load()
# ...Binaraizer code...
del loaded_files
gc.collect()
```

An example of using storage:
```python
import mlup
from mlup import constants

up = mlup.UP(
    conf=mlup.Config(
        storage_type=constants.StorageType.disk,
        storage_kwargs={
            "path_to_files": "/path/to/my/model.onnx",
            # There may be several files in the folder. 
            # To uniquely identify your model file, you can use its full name in the mask.
            "file_mask": "model.onnx",
        },
    )
)
up.ml.load()
```

## Custom storage

If the capabilities of mlup storages are not enough for you, you can write your own storage.

The storage interface is very simple:
```python
# my_module.py
from dataclasses import dataclass
from typing import List, Union
from mlup.constants import LoadedFile
from mlup.ml.storage.base import BaseStorage


@dataclass
class MyStorage(BaseStorage):
    password: str
    login: str = 'default login'
    
    @classmethod
    def load_bytes_single_file(cls, *args, **kwargs) -> Union[str, bytes]:
        pass

    @classmethod
    def load(cls) -> List[LoadedFile]:
        pass
```

Where:

* `load` - method that calls mlup. Everything happens in this method. It also calls `load_bytes_single_file` for each file that needs to be analyzed, loaded into memory or downloaded to disk.
* `load_bytes_single_file` - this method is called inside `load` for each file when the file needs to be parsed, loaded into memory or downloaded to its disk.

And specify the path to import your module in `storage_type="my_module.MyStorage"`. And `storage_kwargs={"password": "password from my storage", "login": "not default login"}`.

**IMPORTANT: a storage written independently must be available for import on the server on which you run the mlup application.**

The easiest way to do this is to create your own python library with your storages and other useful classes and install it on your server along with the pymlup library.
