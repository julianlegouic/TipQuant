import os
import sys
from streamlit.web import cli as stcli

from src.utils import makedirs

if __name__ == '__main__':
    # Clear tmp directory from previous session
    makedirs("tmp", exist_ok=True)
    makedirs(os.path.join("tmp", "primary"), exist_ok=True)
    makedirs(os.path.join("tmp", "secondary"), exist_ok=True)
    sys.argv = ["streamlit", "run", "src/ui/app.py"]
    sys.exit(stcli.main())
