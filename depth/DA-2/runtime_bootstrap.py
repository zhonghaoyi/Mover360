import ctypes
import os
import sys
import sysconfig
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache"
SRC_DIR = APP_DIR / "src"


def ensure_runtime_env():
    CACHE_DIR.mkdir(exist_ok=True)
    mpl_config_dir = CACHE_DIR / "matplotlib"
    mpl_config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    # Make the bundled da2 package importable without a separate pip install.
    src_path = str(SRC_DIR)
    if SRC_DIR.is_dir() and src_path not in sys.path:
        sys.path.insert(0, src_path)


def preload_bundled_cuda_libs():
    purelib = Path(sysconfig.get_paths()["purelib"])
    nvjitlink_lib = purelib / "nvidia" / "nvjitlink" / "lib" / "libnvJitLink.so.12"
    if nvjitlink_lib.exists():
        ctypes.CDLL(str(nvjitlink_lib), mode=ctypes.RTLD_GLOBAL)
