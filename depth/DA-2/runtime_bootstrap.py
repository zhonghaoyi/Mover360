import ctypes
import os
import sysconfig
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache"


def ensure_runtime_env():
    CACHE_DIR.mkdir(exist_ok=True)
    mpl_config_dir = CACHE_DIR / "matplotlib"
    mpl_config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def preload_bundled_cuda_libs():
    purelib = Path(sysconfig.get_paths()["purelib"])
    nvjitlink_lib = purelib / "nvidia" / "nvjitlink" / "lib" / "libnvJitLink.so.12"
    if nvjitlink_lib.exists():
        ctypes.CDLL(str(nvjitlink_lib), mode=ctypes.RTLD_GLOBAL)
