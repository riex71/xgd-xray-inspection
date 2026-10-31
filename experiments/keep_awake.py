"""Prevent system sleep only while the authorized validation supervisor is active; restore on exit."""
import ctypes
import json
import os
import time
from pathlib import Path


def main():
    if os.name!='nt':return
    root=Path(__file__).resolve().parents[1]/'outputs/experiments/resume'
    es=ctypes.windll.kernel32.SetThreadExecutionState
    es(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED; display is allowed to sleep.
    try:
        for _ in range(12*60*2):
            p=root/'STATUS.json'
            if p.exists():
                try:
                    s=json.loads(p.read_text(encoding='utf8'))
                    if s.get('status') in ('COMPLETE','FAILED'):break
                except (OSError,json.JSONDecodeError):pass
            time.sleep(30)
    finally:es(0x80000000)


if __name__=='__main__':main()
