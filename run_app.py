import sys
from streamlit.web import cli as stcli

from src.utils import makedirs

if __name__ == '__main__':
    # Clear tmp directory from previous session
    makedirs("tmp", exist_ok=False)
    makedirs("tmp/raw", exist_ok=False)
    makedirs("tmp/mask", exist_ok=False)
    sys.argv = ["streamlit", "run", "src/ui/app.py"]
    sys.exit(stcli.main())
