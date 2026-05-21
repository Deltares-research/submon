import subprocess
import sys

subprocess.run([sys.executable, "./src/submon/main.py"] + sys.argv[1:])
