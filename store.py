import json
import os
import pickle
import re
import textwrap
from functools import partial

import pandas as pd

from . import base
from .base import Bundle, try_import

cx = try_import("connectorx")
pa, pq = try_import("pyarrow"), try_import("pyarrow.parquet")
duckdb = try_import("duckdb")


class Database:
    def __init__(self, conn, *, sql_path=None, **kwargs):
        assert sql_path is None or sql_path[0] in [".", "/"]
        self.conn = conn
        self.use_cx = type(conn) is str
        self.sql_path = sql_path
        self.render_kwargs = kwargs

    def query(self, sql=None, macro=None, *, debug=None, as_type=None, **kwargs):
        sql = base.render(sql or self.sql_path, macro, **(self.render_kwargs | kwargs))
        sql = textwrap.dedent(sql)
        if debug:
            self._debug(sql, debug)
            if "a" in debug:
                raise Exception("Aborted after rendering")
        try:
            return self._query(sql, as_type)
        except Exception as exc:
            error = self._error(exc)
            if not error:
                raise
        print("Error: " + error.message)
        if error.line_num:
            for num, line in enumerate(sql.split("\n"), 1):
                diff = error.line_num - num
                if abs(diff) <= 3:
                    print(f"{num:3d}{':' if diff else '>'} {line}")

    def _debug(self, sql, flags):
        print(sql)

    def _query(self, sql, as_type):
        if self.use_cx:
            return cx.read_sql(self.conn, sql, return_type=as_type or "pandas")
        else:
            return pd.read_sql(sql, self.conn)

    def _error(self, exc):
        return None


class BigQueryDatabase(Database):
    def __init__(self, *args, bqstorage=True, **kwargs):
        self.bqstorage = bqstorage
        super().__init__(*args, **kwargs)

    def _query(self, sql, as_type):
        if self.use_cx:
            # Has issues reading array columns
            # https://github.com/sfu-db/connector-x/issues/818
            return super()._query(sql, as_type)
        else:
            job = self.conn.query(sql)
            fetch = job.to_arrow if as_type == "arrow" else job.to_dataframe
            return fetch(create_bqstorage_client=self.bqstorage)

    def _error(self, exc):
        pattern = r"message: (.*?(?: at \[(\d+):\d+\]))"
        match = re.search(pattern, exc.args[0], re.DOTALL)
        if match:
            message = match[1].replace(r"\"", '"')
            line_num = match[2] and int(match[2])
            return Bundle(message=message, line_num=line_num)

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

    def put(self, name, value):
        dump, mode = self._serializer(name, value, None)
        with self._open(name, "w" + mode) as f:
            dump(value, f)

    def get(self, name, as_type="pandas"):
        assert as_type in ("pandas", "duckdb", "arrow", "path", None)
        load, mode = self._serializer(name, None, as_type)
        with self._open(name, "r" + mode) as f:
            return load(f)

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
                if as_type and self._type(value) != as_type:
                    value = self.get(name, as_type)
        return value

    def _open(self, name, mode):
        return open(f"{self.path}/{name}", mode)

    def _serializer(self, name, value, as_type):
        ext = os.path.splitext(name)[1]
        if as_type == "path":
            load, mode = lambda f: f.name, ""
        elif ext in [".pickle", ".pkl", ""]:
            load, dump, mode = pickle.load, pickle.dump, "b"
        elif ext == ".json":
            load, dump, mode = partial(json.load, object_hook=Bundle), json.dump, "t"
        elif ext == ".parquet":
            load = {
                "pandas": pd.read_parquet,
                "arrow": pa.parquet.read_table,
                "duckdb": lambda f: duckdb.read_parquet(f.name),
            }.get(as_type)
            dump = {
                "pandas": pd.DataFrame.to_parquet,
                "arrow": pa.parquet.write_table,
                "duckdb": lambda v, f, **kwargs: v.write_parquet(f.name, **kwargs),
            }.get(value is not None and self._type(value))
            dump = dump and partial(dump, compression="snappy")
            mode = "b"

        return load if value is None else dump, mode

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
