import subprocess,sys,json,datetime
from pathlib import Path
out=Path(__file__).resolve().parent
status={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stages':{}}
for name in ['train_base.py','train_sft.py','compare_maps.py']:
    status['stages'][name]={'status':'running'}
    (out/'status.json').write_text(json.dumps(status,indent=2))
    print('Starting',name,flush=True)
    with (out/(name+'.log')).open('w') as log:
        p=subprocess.run([sys.executable,'-u',str(out/name)],cwd=out,stdout=log,stderr=subprocess.STDOUT)
    status['stages'][name]={'status':'completed' if p.returncode==0 else 'failed','returncode':p.returncode}
    (out/'status.json').write_text(json.dumps(status,indent=2))
    print('Finished',name,'exit',p.returncode,flush=True)
    if p.returncode:
        print((out/(name+'.log')).read_text()[-6000:],flush=True)
        sys.exit(p.returncode)
status['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(out/'status.json').write_text(json.dumps(status,indent=2))
print('Completed:',out/'REPORT.md',flush=True)
