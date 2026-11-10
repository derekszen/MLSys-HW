from pathlib import Path
import argparse,hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'homework-2/scripts'))
from summarize_streaming import summarize
from summarize_benchmarks import source_identity
from run_model import project_source_fingerprints
r=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description='Revalidate recorded TP1 streaming results, reference agreement and source binding.')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
ref=json.loads((ROOT/'homework-1/rebuild/results/2026-11-06/reference.json').read_text())
refs={c['name']:c for c in ref['cases']}
expected_source=source_identity({'reproducibility':{'source_trees':project_source_fingerprints()}})
expected_cases=hashlib.sha256((ROOT/'homework-2/cases.json').read_bytes()).hexdigest()
expected_runner=hashlib.sha256((ROOT/'homework-2/scripts/run_model.py').read_bytes()).hexdigest()
records={};summary=[];max_error=0;correct_requests=0;stream_requests=0
for name in ['tp1-c1-stream-smoke','tp1-c1-correctness','tp1-c1-streaming','tp1-c2-correctness','tp1-c2-streaming']:
    path=r/name/'result.json';result=json.loads(path.read_text());records[name]=result
    status=json.loads((r/(name+'-status.json')).read_text())
    assert status['state']=='finished' and status['exit_code']==0,name
    assert result['status']=='generation_completed_not_yet_compared',name
    assert result['cases_sha256']==expected_cases,name
    assert source_identity(result)==expected_source,name
    if result['mode']=='streaming':
        assert result['reproducibility']['runner_sha256']==expected_runner,name
        verified=summarize(result)
        for row in verified['summary']:
            summary.append({'run':name,'concurrency':result['concurrency'],**row})
        stream_requests+=verified['verified_requests']
    for run in result['runs']:
        expected=refs[run['case']]
        assert run['input_tokens']==len(expected['input_ids']),name
        for output in run['outputs']:
            assert output['output_ids']==expected['output_ids'],(name,run['case'])
            if result['mode']=='correctness':
                logs=output['meta_info']['output_token_logprobs']
                assert len(logs)==16 and [x[1] for x in logs]==output['output_ids']
                errors=[abs(a[0]-b) for a,b in zip(logs,expected['selected_logprobs'])]
                assert len(errors)==16 and all(math.isfinite(x) and x<=.15 for x in errors)
                max_error=max(max_error,max(errors));correct_requests+=1
for c in (1,2):
    correct=records[f'tp1-c{c}-correctness'];stream=records[f'tp1-c{c}-streaming']
    for field in ['manifest_sha256','config_sha256','cases_sha256','concurrency']:
        assert correct[field]==stream[field],field
    for field in ['runtime_versions','environment']:
        assert correct['reproducibility'][field]==stream['reproducibility'][field],field
    normalized=lambda x:{k:v for k,v in x['config'].items() if k not in ('stream_interval','enable_output_logprobs')}
    assert normalized(correct)==normalized(stream)
output={'date':'2026-11-09','scope':'New real TP1 streaming validation on physical chip1; TP2 pending free same-card pair','passed':True,'correctness_requests':correct_requests,'streamed_requests_including_smoke':stream_requests,'measured_streaming_requests':stream_requests-2,'reference':'homework-1/rebuild/results/2026-11-06/reference.json','reference_sha256':hashlib.sha256((ROOT/'homework-1/rebuild/results/2026-11-06/reference.json').read_bytes()).hexdigest(),'max_selected_logprob_error':max_error,'source_trees_match_local_deployed_runtime':True,'runtime_and_model_config_matched_within_each_correctness_streaming_pair':True,'summary':summary}
output['inputs_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(r.rglob('result.json'))}
output['verifier_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
with args.output.open('x') as target:target.write(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
