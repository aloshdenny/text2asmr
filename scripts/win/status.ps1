# research server status for scripts/status_all.sh (read-only)
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
'D: free {0:N0} GB, C: free {1:N1} GB' -f ((Get-PSDrive D).Free/1GB), ((Get-PSDrive C).Free/1GB)
foreach ($f in (Get-ChildItem D:\t2a\jobs\*.pid -ErrorAction SilentlyContinue)) {
  $alive = [bool](Get-Process -Id (Get-Content $f) -ErrorAction SilentlyContinue)
  $last = Get-Content ("D:\t2a\logs\" + $f.BaseName + ".log") -Tail 1 -ErrorAction SilentlyContinue
  '{0} job {1}: {2}' -f $(if ($alive) { 'OK  ' } else { 'DONE' }), $f.BaseName, $last
}
exit 0
