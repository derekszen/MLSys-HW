import datetime,json,os,re,socket,subprocess,sys
from pathlib import Path
ROOT=Path('/home/u2501112130/MLSys-HW/hw2-streaming-20261109')
PY='/home/u2501112130/MLSys-HW/.venv/bin/python'
name,mode,concurrency,repeats=sys.argv[1:]
assert re.fullmatch(r'[a-z0-9-]+',name) and mode in ('correctness','streaming')
assert concurrency in ('1','2') and repeats in ('1','3')
status_path=ROOT/'runs'/f'{name}-status.json'
assert not status_path.exists() and not (ROOT/'runs'/name).exists()
status={'state':'starting','started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'physical_devices':[1],'tp':1,'mode':mode,'concurrency':int(concurrency),'repeats':int(repeats),'supervisor_pid':os.getpid()}
def save():status_path.write_text(json.dumps(status,indent=2)+'\n')
save()
try:
    inventory=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True,timeout=20).stdout
    (ROOT/'runs'/f'{name}-inventory.txt').write_text(inventory)
    rows=re.findall(r'^\|\s+(\d+)\s+(\d+)\s+(\d+)\s+',inventory,re.M)
    assert not any(card=='0' and chip=='1' for card,chip,pid in rows),'Chip 1 has a listed workload'
    with socket.socket() as portcheck:portcheck.bind(('0.0.0.0',31891))
    command=['timeout','--signal=TERM','--kill-after=20s','600s',PY,'homework-2/scripts/run_model.py','--tp','1','--physical-devices','1','--model','/home/u2501112130/MLSys-HW/models/Qwen3.5-0.8B','--dtype','bfloat16','--cases','homework-2/cases.json','--output-dir',str(ROOT/'runs'/name),'--memory-fraction','0.12','--concurrency',concurrency,'--mode',mode,'--port','31891','--repeats',repeats]
    status['command']=command;status['state']='running';save()
    with (ROOT/'runs'/f'{name}.log').open('x') as log:
        completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    status['exit_code']=completed.returncode
    status['state']='finished' if completed.returncode==0 else 'failed'
except BaseException as exc:
    status['state']='failed';status['error']=f'{type(exc).__name__}: {exc}'
finally:
    status['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
