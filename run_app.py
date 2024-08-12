import sys
from streamlit import cli as stcli

from src.utils import makedirs

if __name__ == '__main__':
    # Clear temp directory from previous session
    makedirs("temp", exist_ok=False)
    sys.argv = ["streamlit", "run", "src/ui/app.py"]
    sys.exit(stcli.main())
