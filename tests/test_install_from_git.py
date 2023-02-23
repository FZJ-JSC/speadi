import subprocess
import sys


def test_install_SPEADI_from_github():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/FZJ-JSC/speadi.git"])
