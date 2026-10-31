import numpy as np
from experiments.followup import policy as P
from src.followup_safety import resolution_compatible,confirmation_schema,fixed_sequence,pipeline_signature

def box(x,y,s=.9,side=.01):return [x-side/2,y-side/2,x+side/2,y+side/2,s]

def test_high_average_score_cannot_hide_unmatched_low_score_candidate():
    high=[box(.1,.1),box(.4,.4)]
    assert P.residual_flag(high+[box(.8,.8,.12)],high,100,100,.10,.25)
    assert not P.residual_flag(high+[box(.11,.10,.12)],high,100,100,.10,.25)

def test_two_real_close_objects_are_removed_by_4px_but_not_2px():
    b=[box(.5,.5),box(.53,.5,.8)];gt=[x[:4] for x in b]
    four=P.confirmed(b,100,100,{**P.default(),'e0':4})
    two=P.confirmed(b,100,100,{**P.default(),'e0':2})
    assert P.image_eval(four,gt,100,100,.25)['tp3']==1
    assert P.image_eval(two,gt,100,100,.25)['tp3']==2

def test_matching_is_one_to_one_and_miss_is_not_only_low_confidence():
    a=[box(.5,.5),box(.51,.5)];b=[box(.5,.5)]
    assert P.match(a,b,100,100,3)[1:]==(1,0)
    r=P.image_eval(b,[q[:4] for q in a],100,100,.25)
    assert r['actual_missing8']==1 and r['trust']==.9

def test_resolution_unknown_or_stale_signature_blocks_even_similar_network_scale():
    entry={'native_resolutions':[[352,332]],'pipeline_signature':'one'}
    assert resolution_compatible(entry,[352,332],'one')[0]
    assert not resolution_compatible(entry,[316,332],'one')[0]
    assert not resolution_compatible(entry,[352,332],'two')[0]
    assert not resolution_compatible(entry,[True,332],'one')[0]

def test_fixed_sequence_stops_at_first_failure_and_does_not_skip_to_later_success():
    assert fixed_sequence([.01,.2,.001])['passed']==[0]
    assert fixed_sequence([.01,float('nan')])['invalid']

def test_confirmation_is_not_product_pass_or_verified_completeness():
    assert confirmation_schema(False,'ACCEPT')['completeness_status']=='NOT_CERTIFIED'
    assert not confirmation_schema(False,'ACCEPT')['product_pass_certified']
    assert not confirmation_schema(True,'REVIEW')['detections_confirmed']

def test_new_policy_or_source_invalidates_old_calibration_signature():
    a=pipeline_signature({'score':.25},['sha1'],{'source':'v1'})
    assert a!=pipeline_signature({'score':.15},['sha1'],{'source':'v1'})
    assert a!=pipeline_signature({'score':.25},['sha1'],{'source':'v2'})
