import json
import os
import pickle
import re
import textwrap
from functools import partial
from gzip import GzipFile
from pathlib import Path

import pandas as pd

from . import base
from .base import Bundle, try_import

cx = try_import("connectorx")
pq = try_import("pyarrow.parquet")
duckdb = try_import("duckdb")


class Database:
    def __init__(self, conn, *, sql_path=None, **kwargs):
        assert type(sql_path) is not str or sql_path[0] in [".", "/"]
        self.conn = conn
        self.use_cx = type(conn) is str
        self.sql_path = Path(sql_path)
        self.render_kwargs = kwargs

    def query(self, sql=None, macro=None, *, debug=None, as_type=None, **kwargs):
        sql = base.render(sql or self.sql_path, macro, **(self.render_kwargs | kwargs))
        sql = textwrap.dedent(sql)
        if debug and debug != "n":
            self._debug(sql, debug)
            if "a" in debug:
                raise Exception("Aborted after rendering")
        try:
            return self._query(sql, as_type)
        except Exception as exc:
            message, line_num = self._error(exc)
            if not message:
                raise
        print("Error: " + message)
        if line_num:
            for num, line in enumerate(sql.split("\n"), 1):
                diff = line_num - num
                if abs(diff) <= 3:
                    print(f"{num:3d}{':' if diff else '>'} {line}")

    def _query(self, sql, as_type):
        if self.use_cx:
            return_type = {None: "pandas", "duckdb": "arrow"}.get(as_type, as_type)
            value = cx.read_sql(self.conn, sql, return_type=return_type)
            return duckdb.from_arrow(value) if as_type == "duckdb" else value
        else:
            return pd.read_sql(sql, self.conn)

    def _error(self, exc):
        return None, None

    def _debug(self, sql, flags):
        print(sql)


class BigQueryDatabase(Database):
    def __init__(self, *args, bqstorage=True, **kwargs):
        self.bqstorage = bqstorage
        super().__init__(*args, **kwargs)

    def _query(self, sql, as_type):
        if self.use_cx:
            # Has issues reading array columns
            # https://github.com/sfu-db/connector-x/issues/818
            return super()._query(sql, as_type)

        assert as_type in (None, "pandas", "arrow", "duckdb")
        job = self.conn.query(sql)
        fetch = job.to_dataframe if as_type in (None, "pandas") else job.to_arrow
        value = fetch(create_bqstorage_client=self.bqstorage)
        return duckdb.from_arrow(value) if as_type == "duckdb" else value

    def _error(self, exc):
        pattern = r"message: (.*?(?: at \[(\d+):\d+\]))"
        match = re.search(pattern, exc.args[0], re.DOTALL)
        return match and (match[1].replace(r"\"", '"'),  match[2] and int(match[2]))

    def metadata(
        self,
        schema,
        table,
        columns=["data_type", "is_nullable", "is_partitioning_column"],
        **kwargs,
    ):
        return self.query(
            f"""
            SELECT column_name, {", ".join(columns)}
            FROM {schema}.INFORMATION_SCHEMA.COLUMNS
            WHERE table_name = "{table}"
            """,
            **kwargs,
        ).set_index("column_name")


class Storage:
    def __init__(self, path):
        self.path = path

    def put(self, name, value, **kwargs):
        self._io(name, value, None, kwargs)

    def get(self, name, as_type=None, **kwargs):
        return self._io(name, None, as_type, kwargs)

    def get_or_create(self, name, factory, *, cache="use", as_type=None):
        assert cache in ("use", "set", "skip")
        value = None
        if cache == "use":
            try:
                value = self.get(name, as_type)
                print("Loaded from storage")
            except Exception:
                pass
        if value is None:
            value = factory()
            if cache != "skip":
                self.put(name, value)
                # For duckdb always return parquet-based (not arrow-based) rels
                if as_type == "duckdb" or (as_type and self._type(value) != as_type):
                    value = self.get(name, as_type)
        return value

    def _open(self, name, mode):
        return open(f"{self.path}/{name}", mode)

    def _io(self, name, value, as_type, kwargs):
        root, ext = os.path.splitext(name)
        compress = ext in (".gz", ".gzip")
        if compress:
            root, ext = os.path.splitext(root)
        get = value is None
        if ext in (".pickle", ".pkl", ""):
            load, dump = pickle.load, pickle.dump
        elif ext == ".json":
            load, dump = partial(json.load, object_hook=Bundle), json.dump
        elif ext == ".parquet":
            type = (as_type or "pandas") if get else self._type(value)
            assert type in ("pandas", "arrow", "duckdb")
            kwargs = ({} if get else {"compression": "snappy"}) | kwargs
            if type == "pandas":
                load, dump = pd.read_parquet, pd.DataFrame.to_parquet
            elif type == "arrow":
                load, dump = pq.read_table, pq.write_table
            elif type == "duckdb":
                fun = duckdb.read_parquet if get else value.write_parquet
                return fun(f"{self.path}/{name}", **kwargs)

        mode = ("r" if get else "w") + ("t" if ext == ".json" else "b")
        fun = load if get else partial(dump, value)
        with self._open(name, mode) as file:
            if compress:
                with GzipFile(fileobj=file, mode=mode) as gzip_file:
                    return fun(gzip_file, **kwargs)
            return fun(file, **kwargs)

    def _type(self, value):
        return {
            "DataFrame": "pandas",
            "Table": "arrow",
            "DuckDBPyRelation": "duckdb",
        }[value.__class__.__name__]


class GoogleStorage(Storage):
    def __init__(self, client, path, *, local_path=None):
        self.client = client
        bucket, self.path = path.rstrip("/").split("/", 1)
        self.bucket = self.client.bucket(bucket)
        self.local_cache = Storage(local_path) if local_path else None

    def get(self, name, as_type=None, **kwargs):
        value = super().get(name, "arrow" if as_type == "duckdb" else as_type, **kwargs)
        return duckdb.from_arrow(value) if as_type == "duckdb" else value

    def get_or_create(self, name, factory, *, cache="use"):
        fun = partial(super().get_or_create, name, factory, cache=cache)
        if self.local_cache:
            return self.local_cache.get_or_create(name, fun, cache=cache)
        else:
            return fun()

    def remove(self, name):
        self._blob(name).delete()

    def list(self):
        for name, blob in self:
            size = blob.size / 1024**2
            print(f"{name}: {size:.2f} MB")

    def __iter__(self):
        for blob in self.bucket.list_blobs(prefix=self.path):
            yield blob.name.removeprefix(self.path + "/"), blob

    def _open(self, name, mode):
        return self._blob(name).open(mode)

    def _blob(self, name):
        return self.bucket.blob(f"{self.path}/{name}")
