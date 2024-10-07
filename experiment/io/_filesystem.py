import os
import shutil

import s3fs


class FileSystem:
    def copy(self, lpath, rpath):
        raise NotImplementedError

    def makedirs(self, path):
        raise NotImplementedError

    def open(self, path: str, mode: str):
        raise NotImplementedError

    def get_path(self, path: str):
        raise NotImplementedError

    def exists(self, path: str):
        raise NotImplementedError


class S3FileSystem(FileSystem):
    def __init__(
        self,
        anon: bool = False,
        endpoint: str = None,
        endpoint_env_var: str = "AWS_S3_ENDPOINT",
        key_env_var: str = "AWS_ACCESS_KEY_ID",
        secret_env_var: str = "AWS_SECRET_ACCESS_KEY",
        token_env_var: str = "AWS_SESSION_TOKEN",
    ):
        if endpoint is None:
            endpoint = os.environ[endpoint_env_var]

        kwargs = {"anon": anon, "endpoint": endpoint}
        if os.environ.get(key_env_var) is not None:
            kwargs["key"] = os.environ[key_env_var]
        if os.environ.get(secret_env_var) is not None:
            kwargs["secret"] = os.environ[secret_env_var]
        if os.environ.get(token_env_var) is not None:
            kwargs["token"] = os.environ[token_env_var]

        self._fs = s3fs.S3FileSystem(**kwargs)

    def copy(self, lpath, rpath):
        rpath = f"{rpath}/"
        self._fs.put(lpath, rpath, recursive=True)

    def makedirs(self, path):
        self._fs.makedirs(path, exist_ok=True)

    def open(self, path: str, mode: str):
        f = self._fs.open(path, mode=mode)
        return f

    def get_path(self, path: str):
        return s3fs.S3Map(root=f"s3://{path}", s3=self._fs)

    def exists(self, path: str):
        return self._fs.exists(path)


class LocalFileSystem(FileSystem):
    def copy(self, lpath, rpath):
        rpath = f"{rpath}/"
        shutil.copytree(lpath, rpath)

    def makedirs(self, path):
        os.makedirs(path, exist_ok=True)

    def open(self, path: str, mode: str):
        f = open(path, mode=mode)
        return f

    def get_path(self, path: str):
        return path

    def exists(self, path: str):
        return os.path.exists(path)
