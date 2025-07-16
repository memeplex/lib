import json
import os
import pickle
import re
import textwrap
from functools import partial

import pandas as pd

from . import base
from .base import Bundle


class Database:
    def __init__(self, conn, *, sql_path=None, **kwargs):
        assert sql_path is None or sql_path[0] in [".", "/"]
        self.conn = conn
        self.use_cx = type(conn) is str
        self.sql_path = sql_path
        self.render_kwargs = kwargs

    def query(self, sql=None, macro=None, *, debug=False, progress=False, **kwargs):
        sql = base.render(sql or self.sql_path, macro, **(self.render_kwargs | kwargs))
        sql = textwrap.dedent(sql)
        if debug and debug != "n":
            self._debug(sql, debug)
            if "a" in debug:
                raise Exception("Aborted after rendering")
        try:
            return self._query(sql, progress)
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

    def _debug(self, sql, mode):
        print(sql)

    def _query(self, sql, progress):
        if self.use_cx:
            import connectorx

            return connectorx.read_sql(self.conn, sql)
        else:
            return pd.read_sql(sql, self.conn)

    def _error(self, exc):
        return None


class BigQueryDatabase(Database):
    def _query(self, sql, progress):
        if self.use_cx:
            # Has issues reading array columns
            # https://github.com/sfu-db/connector-x/issues/818
            return super()._query(sql, progress)
        else:
            import pandas_gbq

            return pandas_gbq.read_gbq(
                sql,
                bigquery_client=self.conn,
                progress_bar_type="tqdm" if progress else None,
            )

    def _error(self, exc):
        pre = r'message: "' if self.use_cx else r"https:[^ ]+"
        post = r'", status:' if self.use_cx else r"$"
        pattern = rf"{pre}(.*?(?: at \[(\d+):\d+\])?){post}"
        match = re.search(pattern, exc.args[0])
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
        serializer = self._serializer(name)
        with self._open(name, "w" + serializer.mode) as f:
            serializer.dump(value, f)

    def get(self, name):
        serializer = self._serializer(name)
        with self._open(name, "r" + serializer.mode) as f:
            return serializer.load(f)

    def get_or_create(self, name, factory, *, cache="use"):
        assert cache in ("use", "set", "skip")
        value = None
        if cache == "use":
            try:
                value = self.get(name)
                print("Loaded from storage")
            except Exception:
                pass
        if value is None:
            value = factory()
            if cache != "skip":
                self.put(name, value)
        return value

    def _open(self, name, mode):
        return open(f"{self.path}/{name}", mode)

    def _serializer(self, name):
        ext = os.path.splitext(name)[1]
        if ext in [".pickle", ".pkl", ""]:
            load, dump, mode = pickle.load, pickle.dump, "b"
        elif ext == ".json":
            load, dump, mode = partial(json.load, object_hook=Bundle), json.dump, "t"
        return Bundle(load=load, dump=dump, mode=mode)


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
