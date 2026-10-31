"""Batch-one end-to-end timing, including raw read/preprocessing/decision; no GT or recipe fitting."""
import datetime
import time
import importlib.metadata as md

import numpy as np
import torch

from experiments import lib as L


def main():
    report={'time':datetime.datetime.now().astimezone().isoformat(),
            'scope':'loaded model, one warmup then first 20 sorted development images per profile; batch1',
            'includes':'raw file read, preprocessing, GPU inference, E0, quality/calibration/decision',
            'excludes':'model load, human review, acquisition, factory integration',
            'source_sha256':L.X.sha256_file(L.ROOT/'experiments/operational_timing.py'),
            'GPU':torch.cuda.get_device_name(0),'torch':md.version('torch'),
            'evaluation79_label_reads':0,'profiles':{}}
    for prof in ('dfl_src12','dfl_src3_2'):
        an=L.X.XGDAnalyzer(L.CFG,prof,17)
        stems=sorted(L.targets(prof))[:20]
        times=[]; decisions=[]
        for i,s in enumerate([stems[0],*stems]):
            m={'calibration_id':L.META[s]['device'],'session':L.META[s]['date'],'preprocessed':False}
            torch.cuda.synchronize(); start=time.perf_counter()
            r=an.analyze(image_path=L.XC.raw_path(L.CFG,s),metadata=m)
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            if i:times.append(elapsed);decisions.append(r['decision'])
        report['profiles'][prof]={'stems':stems,'seconds':times,'decisions':decisions,
                                 'mean_seconds':float(np.mean(times)),
                                 'P50_P90_P95_seconds':np.percentile(times,[50,90,95]).tolist()}
        del an
    L.jdump(L.OUT/'resume/OPERATIONAL_TIMING.json',report)
    print('operational end-to-end timing complete; no human review measured',flush=True)


if __name__=='__main__':main()
