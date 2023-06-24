from collections import OrderedDict
from threading import RLock
import time


class TTLOrderedDict(OrderedDict):
    """
    OrderedDict with TTL
    Extra args and kwargs are passed to initial .update() call
    """
    def __init__(self, default_ttl, *args, **kwargs):
        """
        Be warned, if you use this with Python versions earlier than 3.6
        when passing **kwargs order is not preseverd.
        """
        assert isinstance(default_ttl, int)
        self._default_ttl = default_ttl
        self._lock = RLock()
        super().__init__()
        self.update(*args, **kwargs)

    def __repr__(self):
        return '<TTLOrderedDict@%#08x; ttl=%r, OrderedDict=%r;>' % (
            id(self), self._default_ttl, self.items())

    def __len__(self):
        with self._lock:
            self._purge()
            return super().__len__()

    def set_ttl(self, key, ttl, now=None):
        """Set TTL for the given key"""
        if now is None:
            now = time.time()
        with self._lock:
            value = self[key]
            super().__setitem__(key, (now + ttl, value))

    def get_ttl(self, key, now=None):
        """Return remaining TTL for a key"""
        if now is None:
            now = time.time()
        with self._lock:
            expire, _value = super().__getitem__(key)
            return expire - now

    def expire_at(self, key, timestamp):
        """Set the key expire timestamp"""
        with self._lock:
            value = self.__getitem__(key)
            super().__setitem__(key,  (timestamp, value))

    def is_expired(self, key, now=None):
        """ Check if key has expired, and return it if so"""
        with self._lock:
            if now is None:
                now = time.time()

            expire, _value = super().__getitem__(key)

            if expire:
                if expire < now:
                    return key

    def _purge(self):
        _keys = list(super().__iter__())
        _remove = [key for key in _keys if self.is_expired(key)]  # noqa
        [self.__delitem__(key) for key in _remove]

    def __iter__(self):
        """
        Yield only non expired keys, without purging the expired ones
        """
        with self._lock:
            for key in super().__iter__():
                if not self.is_expired(key):
                    yield key

    def __setitem__(self, key, value):
        with self._lock:
            if self._default_ttl is None:
                expire = None
            else:
                expire = time.time() + self._default_ttl
            super().__setitem__(key,  (expire, value))

    def __delitem__(self, key):
        with self._lock:
            super().__delitem__(key)

    def __getitem__(self, key):
        with self._lock:
            if self.is_expired(key):
                self.__delitem__(key)
                raise KeyError
            item = super().__getitem__(key)[1]
            return item

    def keys(self):
        with self._lock:
            self._purge()
            return super().keys()

    def items(self):
        with self._lock:
            self._purge()
            _items = list(super(OrderedDict, self).items())
            return [(k, v[1]) for (k, v) in _items]

    def values(self):
        with self._lock:
            self._purge()
            _values = list(super(OrderedDict, self).values())
            return [v[1] for v in _values]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
