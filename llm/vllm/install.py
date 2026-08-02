import subprocess
import sys
import urllib.request
import json

release_url = "https://api.github.com/repos/vllm-project/vllm-metal/releases/latest"
release = json.load(urllib.request.urlopen(release_url))
wheel_url = next(
    asset["browser_download_url"]
    for asset in release["assets"]
    if asset["name"].endswith(".whl")
)

print(wheel_url)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", wheel_url])