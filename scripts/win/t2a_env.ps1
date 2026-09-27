# text2asmr workspace on D: -- nothing may write to C: (it is full) and nothing touches WSL (superfloat lives there)
$env:TEMP = "D:\t2a\tmp"; $env:TMP = "D:\t2a\tmp"
$env:HF_HOME = "D:\t2a\hf"; $env:HF_HUB_DISABLE_XET = "1"; $env:PYTHONUNBUFFERED = "1"
$env:UV_CACHE_DIR = "D:\t2a\cache\uv"; $env:PIP_CACHE_DIR = "D:\t2a\cache\pip"
$env:UV_PYTHON_INSTALL_DIR = "D:\t2a\python"; $env:TORCH_HOME = "D:\t2a\cache\torch"; $env:XDG_CACHE_HOME = "D:\t2a\cache"
$env:Path = "D:\t2a\bin;D:\t2a\venv\Scripts;" + $env:Path
