import ctypes,json,time,os
from experiments.followup import common as F
def main():
    if os.name!='nt':return
    f=ctypes.windll.kernel32.SetThreadExecutionState;f(0x80000001)
    try:
        for _ in range(1440):
            p=F.OUT/'STATUS.json'
            if p.exists() and json.loads(p.read_text(encoding='utf8')).get('status') in ('FAILED','COMPLETE'):break
            time.sleep(30)
    finally:f(0x80000000)
if __name__=='__main__':main()
