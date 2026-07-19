"""Drive the CLI through a pty so a demo can be recorded as a real session.

Piping stdin into the CLI would produce a transcript with no typing and no
prompts, which is not what a user sees. This spawns the real program on a
pseudo-terminal and types into it, so the recording is the actual interface
behaving normally -- nothing is staged or edited afterwards.

    asciinema rec --overwrite -c "python scripts/record_demo.py" docs/demo.cast
    agg --idle-time-limit 1.5 docs/demo.cast docs/demo.gif
"""

import fcntl
import os
import pty
import select
import struct
import sys
import termios
import time

# Must match the canvas the recording is rendered on, or the CLI wraps at one
# width and the player re-wraps at another.
TERMINAL_SIZE = (24, 80)  # rows, columns

COMMANDS = [
    "load .eval-corpus/spring-petclinic/src/main",
    "Which endpoint creates a new pet owner?",
    "What HTTP method does it use?",
    "exit",
]

PROMPT = b"You: "
TYPING_DELAY = 0.04
PAUSE_BEFORE_TYPING = 0.7
PAUSE_BEFORE_RETURN = 0.35


def type_into(fd: int, text: str) -> None:
    for character in text:
        os.write(fd, character.encode())
        time.sleep(TYPING_DELAY)


def main() -> int:
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp("code-agent", ["code-agent"])
        return 1  # only reached if exec fails

    rows, columns = TERMINAL_SIZE
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))

    pending = list(COMMANDS)
    buffer = b""

    while True:
        try:
            readable, _, _ = select.select([fd], [], [], 0.2)
        except OSError:
            break

        if fd not in readable:
            continue

        try:
            data = os.read(fd, 4096)
        except OSError:
            break
        if not data:
            break

        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()

        buffer += data
        if pending and buffer.endswith(PROMPT):
            time.sleep(PAUSE_BEFORE_TYPING)
            type_into(fd, pending.pop(0))
            time.sleep(PAUSE_BEFORE_RETURN)
            os.write(fd, b"\n")
            buffer = b""

    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


if __name__ == "__main__":
    sys.exit(main())
