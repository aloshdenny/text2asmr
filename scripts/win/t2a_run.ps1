# Detached job runner for the research server: the Windows stand-in for "tmux new -d".
#   powershell -File D:\t2a\t2a_run.ps1 -Name <job> -Cmd "<python args>"
# The job gets its own env (D:-only caches, the t2a venv), survives SSH disconnects (WMI-created, outside the
# session's job object), and logs to D:\t2a\logs\<job>.log. One job per name: refuses if it is already running.
param([Parameter(Mandatory)][string]$Name, [Parameter(Mandatory)][string]$Cmd)
New-Item -ItemType Directory -Force -Path D:\t2a\logs, D:\t2a\jobs | Out-Null
$pidf = "D:\t2a\jobs\$Name.pid"
if ((Test-Path $pidf) -and (Get-Process -Id (Get-Content $pidf) -ErrorAction SilentlyContinue)) { "job $Name already running (pid $(Get-Content $pidf))"; exit 1 }
$wrap = "D:\t2a\jobs\$Name.cmd"
@"
@echo off
set PYTHONUTF8=1& set TEMP=D:\t2a\tmp& set TMP=D:\t2a\tmp& set HF_HOME=D:\t2a\hf& set HF_HUB_DISABLE_XET=1& set PYTHONUNBUFFERED=1
set UV_CACHE_DIR=D:\t2a\cache\uv& set PIP_CACHE_DIR=D:\t2a\cache\pip& set TORCH_HOME=D:\t2a\cache\torch& set XDG_CACHE_HOME=D:\t2a\cache
set PYTHONPATH=D:\t2a\text2asmr& set PATH=D:\t2a\venv\Scripts;D:\t2a\bin;%PATH%
for /f "usebackq delims=" %%L in ("D:\t2a\secrets.env") do set "%%L"
cd /d D:\t2a\text2asmr
echo [%date% %time%] START $Name >> D:\t2a\logs\$Name.log
python $Cmd >> D:\t2a\logs\$Name.log 2>&1
echo [%date% %time%] EXIT %ERRORLEVEL% >> D:\t2a\logs\$Name.log
"@ | Set-Content -Encoding ASCII $wrap
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine="cmd /c `"$wrap`""; CurrentDirectory="D:\t2a"}
if ($r.ReturnValue -ne 0) { "launch failed rc=$($r.ReturnValue)"; exit 1 }
$r.ProcessId | Set-Content $pidf
"started $Name pid $($r.ProcessId) -> D:\t2a\logs\$Name.log"
