import datetime
import hashlib
import json
import os
import subprocess
import time


class Bundle(dict):
    __slots__ = ()

    def __call__(self, *args, **kwargs):
        base = {n: self[n] for n in ([] if not args[0] else args)} if args else self
        return self.__class__({**base, **kwargs})

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return getattr(super(), name)

    def __setattr__(self, name, value):
        self[name] = value

    def __hasattr__(self, name):
        return name in self


class Config(Bundle):
    def uid(self):
        text = json.dumps(self, sort_keys=True)
        return hashlib.blake2b(text.encode('utf-8'), digest_size=16).hexdigest()


def sh(cmd, input=None, capture=True, text=True, check=True):
    if input and not isinstance(input, (bytes, str)):
        input = "\n".join(input)
    return subprocess.run(
        cmd, input=input, capture_output=capture, text=text, check=check, shell=True
    ).stdout


def sh_quote(s):
    return "'" + s.replace("'", "'\\''") + "'"


def render(
    text,
    macro=None,
    *,
    paths=["."],
    filters={},
    globals={},
    env_kwargs={},
    **kwargs,
):
    import jinja2

    loader = jinja2.FileSystemLoader(paths)
    env_kwargs = dict(lstrip_blocks=True, trim_blocks=True) | env_kwargs
    env = jinja2.Environment(loader=loader, **env_kwargs)
    env.filters.update(filters)
    env.globals.update(globals)
    if text[0] in [".", "/"]:
        with open(text) as file:
            text = file.read()
    template = env.from_string(text)
    if macro:
        return getattr(template.module, macro)(**kwargs)
    else:
        return template.render(**kwargs)


def asyn(fun, *args, **kwargs):
    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fun, *args, **kwargs)
    executor.shutdown(wait=False)
    return future


def dt(t=None, tz=None):
    if type(t) in (int, float):
        return datetime.datetime.fromtimestamp(t, tz)
    elif type(t) is str:
        return datetime.datetime.fromisoformat(t)
    elif type(t) is tuple:
        return datetime.datetime(*t, tzinfo=tz)
    elif t is None:
        return datetime.datetime.now(tz)


def iso(dt, format="D T"):
    return dt.strftime(format.replace("D", r"%Y-%m-%d").replace("T", r"%H:%M:%S"))


def span(**kwargs):
    return datetime.timedelta(**kwargs)


def set_tz(tz):
    os.environ["TZ"] = tz
    time.tzset()
