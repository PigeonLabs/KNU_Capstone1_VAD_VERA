import unittest
import numpy as np
from scripts.vera_stage8_eval import refine,metric
from ipad.vera import refine_scores
class Stage8Tests(unittest.TestCase):
 def test_binary_compatibility(self):
  for length in (1,9,16,17,239,600):
   n=(length+15)//16;r=np.random.default_rng(0);x=r.integers(0,2,n);f=r.normal(size=(n,12))
   a=refine(x,f,length);b,_=refine_scores(x,f,length)
   for k in a:np.testing.assert_array_equal(a[k],b[k])
 def test_continuous_and_ties(self):
  self.assertEqual(metric([0,1],[.3,.3])['auroc'],50)
  self.assertEqual(metric([0,1],[.2,.3])['auroc'],100)
  self.assertEqual(metric([0,0],[.2,.3])['auroc'],None)
  self.assertEqual(len(refine([.23,.91],[[1,0],[0,1]],17)['initial']),17)
 def test_nonfinite_rejected(self):
  with self.assertRaises(AssertionError):refine([np.nan],[[1]],1)

class Stage8InferenceEvidenceTests(unittest.TestCase):
 def test_prompt_preserves_official_questions(self):
  from scripts.vera_stage8_common import T1
  from ipad.vera import QUESTIONS
  self.assertIn(QUESTIONS,T1);self.assertEqual(T1.count('<image>'),8);self.assertTrue(T1.endswith('Answer:'))
 def test_same_forward_capture_and_no_retry(self):
  from pathlib import Path
  source=(Path(__file__).resolve().parents[1]/'scripts/vera_stage8_engine.py').read_text()
  self.assertIn('register_forward_hook',source);self.assertIn("'max_new_tokens':1",source)
  self.assertIn("generated[0][0]==captured[0]['vocabulary_argmax']",source)
 def test_zero_diff_bootstrap_has_zero_ci(self):
  from scripts.vera_stage8_eval import paired_bootstrap
  rr=[]
  for scene in ['R01','R02','R03','R04']:
   for v in range(2):
    for c in ['A','B']:
     for y in [0,1]:rr.append(dict(condition=c,scene=scene,original_split='testing',video=str(v),frame=y,label=y,initial=.2,retrieved=.2,smoothed=.2,final=.2))
  b=paired_bootstrap(rr,'A','B',n=20);self.assertEqual(b['ci95']['initial']['auroc'],[0,0]);self.assertEqual(b['undefined_single_class'],0)

class Stage8HybridTests(unittest.TestCase):
 def test_memory_train_excludes_validation_evaluation(self):
  import json
  from pathlib import Path
  root=Path(__file__).resolve().parents[1]
  rr=json.loads((root/'experiments/stage2_1/split.json').read_text())['records']
  train={r['id'] for r in rr if r['split']=='train' and r['video_label']==0}
  hold={r['id'] for r in rr if r['split'] in ['validation','evaluation']}
  self.assertFalse(train & hold)
  for p in (root/'experiments/stage8/dino').glob('*/memory_inputs.json'):
   d=json.loads(p.read_text());self.assertTrue(set(d['normal_train_ids'])<=train);self.assertTrue({r['video_id'] for r in d['selected']}<=train)
 def test_score_explanation_calls_are_separate(self):
  from pathlib import Path
  root=Path(__file__).resolve().parents[1]
  score=(root/'scripts/vera_stage8_engine.py').read_text();explain=(root/'scripts/explain_vera_stage8.py').read_text()
  self.assertIn("'max_new_tokens':1",score);self.assertIn("'max_new_tokens':256",explain);self.assertIn("'repetition_penalty':1.1",explain)
