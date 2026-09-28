#!/usr/bin/env python3
"""Block until a file exists (a stage marker from another job), then exit 0. Used to chain detached jobs."""
import os, sys, time
while not os.path.exists(sys.argv[1]):
    time.sleep(60)
print(f"found {sys.argv[1]}", flush=True)
