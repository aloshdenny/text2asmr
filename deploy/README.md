# Running the dataset build as a resumable service

The build checkpoints every source file to `out/state.json`, so an interrupted
run costs only the file that was in flight. Running it under systemd rather
than tmux means a reboot no longer leaves the job dead until someone notices
(which cost ~1h20m of idle time on 2026-09-02).

    cp t2a-build.service ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now t2a-build.service

`loginctl enable-linger $USER` must be set, or user services stop at logout and
do not start at boot -- which would defeat the point.

Verified behaviour:

- `kill -9` on the main PID -> systemd restarts it, and it resumes from the
  checkpoint rather than starting over
- a clean exit 0 -> `RestartPreventExitStatus=0` stops the unit instead of
  looping a finished job every `RestartSec`

Check on it with:

    systemctl --user status t2a-build.service
    tail -3 ~/t2a/build.log

Do not run this concurrently with a tmux copy: two processes writing the same
`metadata.jsonl` and FLAC paths will corrupt the output.
