# Local process control

Use the native shell to inspect and stop programs you started. Available commands
include `ps`, `pgrep`, `pkill`, the Bash `kill` builtin, `setsid`, and `timeout`.
They operate in this container's PID namespace. No host PID namespace or Docker
socket is exposed. These are shell utilities, not additional benchmark tools.

## Inspect before stopping

```bash
ps -eo pid,ppid,pgid,sid,nlwp,stat,etime,args
pgrep -af '^python3 scratch/my_job.py$'
```

`exec_command` session IDs are not OS PIDs. Read the process list to identify a
PID or process group. Avoid broad `pkill -f` patterns that can match another job
or your own shell command. Do not assume that `command; echo done` succeeded:
check the command's exit status and then check whether the processes remain.

## Run an independently cancellable job

For a program with children, create a separate session/process group and save its
PGID. This example uses Python's standard library; replace the program path.

```bash
mkdir -p scratch
python3 - <<'PY'
import pathlib
import subprocess

with open('scratch/my_job.log', 'ab', buffering=0) as log:
    process = subprocess.Popen(
        ['python3', 'scratch/my_job.py'],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
pathlib.Path('scratch/my_job.pgid').write_text(str(process.pid) + '\n')
print('PGID:', process.pid)
PY
```

The new session leader's PID is also its PGID. Children normally inherit that
group; a child that explicitly starts another session/group must be handled
separately. Recheck saved IDs before reuse because OS process IDs can be recycled.

Inspect the saved group from Bash:

```bash
job_pgid=$(cat scratch/my_job.pgid)
ps -eo pid,ppid,pgid,sid,stat,args | awk -v g="$job_pgid" 'NR == 1 || $3 == g'
```

After verifying that it is the intended group (and not the shell/agent group),
request termination of the whole group, including children:

```bash
kill -TERM -- "-$job_pgid"
```

Allow the program to exit and inspect again. If live processes remain in the
verified group, force termination and verify once more:

```bash
kill -KILL -- "-$job_pgid"
```

For a single confirmed PID, use `kill -TERM PID`, then `kill -KILL PID` only if
needed. `pkill -TERM -g "$job_pgid"` also targets an exact process group; exit
status 1 means no process matched. Zombies (`STAT` starting with `Z`) have exited
and must be reaped by their parent; additional signals do not remove them.

## Interactive cancellation and optional limits

If you want Ctrl-C through `write_stdin`, launch `exec_command` with `tty=true`.
Once a non-TTY session reports closed stdin, `write_stdin` cannot interrupt it;
use another shell call to send signals to its confirmed PID or PGID.

For a command you choose to limit, `timeout --kill-after=5s 60s python3
scratch/my_job.py` requests termination after 60 seconds and escalates five
seconds later if needed. Exit status 124 indicates the timeout; 137 can indicate
SIGKILL. This is an optional local limit, not an imposed benchmark round limit.
