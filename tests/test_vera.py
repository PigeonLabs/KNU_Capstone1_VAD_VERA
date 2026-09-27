import numpy as np
import pytest
from ipad.vera import VeraConfig, build_prompt, parse_response, refine_scores, segments

@pytest.mark.parametrize('length',[1,7,16,17,191,300,409])
def test_full_frame_coverage_and_window_bounds(length):
    ss=segments(length)
    assert [f for s in ss for f in range(s['center'],s['score_end'])]==list(range(length))
    assert all(len(s['frame_ids'])==8 and all(0<=i<length for i in s['frame_ids']) for s in ss)

@pytest.mark.parametrize('text,expected',[('Answers: no\nOutput:\n0',0),('```\nOutput: 1\n```',1)])
def test_strict_terminal_output(text,expected):
    assert parse_response(text)==expected

@pytest.mark.parametrize('text',['No anomaly','Output: 0 or 1','Output: 1 because unusual','Output: 2','I saw 10 things'])
def test_malformed_output_is_not_a_prediction(text):
    with pytest.raises(ValueError):parse_response(text)

def test_short_sequence_smoothing_and_constant_zero():
    for length in [1,17,32]:
        h=len(segments(length));result,_=refine_scores(np.zeros(h),np.ones((h,2)),length)
        assert all(v.shape==(length,) and np.all(v==0) for v in result.values())

def test_paper_temperature_divides():
    config=VeraConfig(retrieval_fraction=1,kernel_size=1)
    output,details=refine_scores([0,1],np.eye(2),32,config)
    expected=1/(np.exp(.1)+1)
    assert output['retrieved'][0]==pytest.approx(expected)
    assert output['retrieved'][16]==pytest.approx(1-expected)
    assert details['neighbors'][0]==[0,1]

def test_invalid_features_rejected():
    with pytest.raises(ValueError):refine_scores([0],np.zeros((1,2)),5)

def test_paper_one_based_position_weights():
    scores,_=refine_scores([1],np.ones((1,2)),16,VeraConfig(kernel_size=1))
    assert scores['final'][7]==1
    assert scores['final'][0]==pytest.approx(np.exp(-49/128))

def test_prompt():
    prompt=build_prompt('$Data\nBased on the analysis above')
    assert prompt.count('<image>')==8
    assert '5. Are there any unusual sounds' in prompt


def test_join_uses_scene_video_and_frame(monkeypatch,tmp_path):
    import csv
    from scripts import evaluate_vera as evaluator
    monkeypatch.setattr(evaluator,'ROOT',tmp_path)
    rows=[]
    for scene in ['R01','R02','R03','R04']:
        rows.extend([dict(scene=scene,video='01',frame=i,label=i%2,final=float(i%2)) for i in range(3)])
        for folder,suffix,columns in [
            ('stage1_reproduction','evaluation/scores.csv',['negative_psnr_with_phase']),
            ('stage2_dinov2','reconstruction/scores.csv',['dino_patch6_with_phase','dino_patch12_with_phase','dino_cls_with_phase','dino_multilevel_with_phase'])]:
            path=tmp_path/'experiments'/folder/scene/suffix
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=['video','frame','label',*columns]);writer.writeheader()
                for i in [1,2,3]:
                    writer.writerow(dict(video='01',frame=i,label=i%2,**{c:float(i%2) for c in columns}))
    result=evaluator.compare(rows)
    assert result['IPAD']['frames']==8
    assert result['IPAD']['vera']['pooled']['auroc']==100
    assert result['DINOv2_reconstruction']['frames']==8


def test_duplicate_journal_is_not_silently_reused(tmp_path):
    from scripts.run_vera import load_records
    p=tmp_path/'journal.jsonl'
    p.write_text('{"center":0}\n{"center":0}\n')
    with pytest.raises(ValueError,match='Duplicate'):load_records(p)
