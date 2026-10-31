"""Prospective deterministic selection; tolerances are never fitted to the follow-up results."""
import copy,math
import numpy as np
from experiments.followup import policy as P,analyze as A

DESIGN='group_date_calibration'

def seeds(report):return list(report['per_seed'].values())
def mean(report,key):return float(np.mean([x[key] for x in seeds(report)]))
def decision(report,key,diagnostic=False):
    label='QUALIFIED_FIXED_DIAGNOSTIC' if diagnostic else DESIGN
    return float(np.mean([x['decisions'][label][key] for x in seeds(report)]))

def clean_gate(control,candidate):
    reasons=[]
    for key in ('R3','F13'):
        if mean(candidate,key)<mean(control,key)-.005-1e-12:reasons.append(key+' mean loss > .005')
    if min(x['R3'] for x in seeds(candidate))<min(x['R3'] for x in seeds(control))-.01-1e-12:reasons.append('worst seed R3 loss > .01')
    if mean(candidate,'FP_per_image3')>mean(control,'FP_per_image3')+.02+1e-12:reasons.append('FP/image increase > .02')
    return reasons

def cost(report):
    # Every condition has equal weight; images/seeds are repetitions, not independent evidence.
    return float(np.mean([decision(report[c],'assumed_cost_50') for c in A.SCREEN_CONDS]))

def policy_screen(base,candidate,config):
    reasons=clean_gate(base['dev_clean'],candidate['dev_clean'])
    if decision(candidate['dev_clean'],'coverage')+1e-12<.5*decision(base['dev_clean'],'coverage'):reasons.append('clean coverage < .5 baseline')
    for c in A.SCREEN_CONDS:
        for metric in ('joint_risk','FN8_joint_risk'):
            if decision(candidate[c],metric)>decision(base[c],metric)+1e-12:reasons.append(c+': '+metric+' increased')
    b=cost(base);v=cost(candidate)
    if v>b*.95+1e-12:reasons.append('primary cost improvement < 5%')
    if config['id'].startswith('Q_'):
        for c,r in candidate.items():
            if c.startswith('STRESS_') and decision(r,'joint_risk')>.05+1e-12:reasons.append(c+': stress risk > .05')
        for cand,ctrl in zip(seeds(candidate['dev_clean']),seeds(base['dev_clean'])):
            ar=cand['decisions'][DESIGN]['conditional_risk'];br=ctrl['decisions'][DESIGN]['conditional_risk']
            if isinstance(ar,(int,float)) and isinstance(br,(int,float)) and ar>br+.02+1e-12:reasons.append('clean conditional risk +.02')
    if config['id'].startswith('GRADE_'):
        for c in A.SCREEN_CONDS:
            if decision(candidate[c],'joint_risk')>.05+1e-12:reasons.append(c+': grade pooled risk > .05')
            if any(x['decisions'][DESIGN]['joint_risk']>.08+1e-12 for x in seeds(candidate[c])):reasons.append(c+': grade seed risk > .08')
    return {'admissible':not reasons,'reasons':reasons,'cost50':v,'baseline_cost50':b,
            'diagnostic_only_cost50':float(np.mean([decision(candidate[c],'assumed_cost_50',True) for c in A.SCREEN_CONDS]))}

def family_key(report):
    free=('R2f_free_M1','R2f_free_M2','R2f_free_M2v2')
    moved=('R2a_move_M1_0','R2a_move_M2_0','R2a_same_M2_0')
    return (float(np.mean([mean(report[c],'actual_missing8')/mean(report[c],'objects') for c in free])),
            -float(np.mean([mean(report[c],'R3') for c in moved])),mean(report['dev_clean'],'FP_per_image3'))

def choose(comparison):
    values={c['id']:c for c in P.candidates()};C=comparison['candidates'];profiles={};tables={}
    for target in A.PROFILES:
        source=next(p for p in A.PROFILES if p!=target);base=C['dfl/P0'][source];table={}
        for ID,cfg in values.items():
            if ID=='P0':continue
            table[ID]=policy_screen(base,C['dfl/'+ID][source],cfg)
        valid=[ID for ID,r in table.items() if r['admissible']]
        ID=min(valid,key=lambda k:(table[k]['cost50'],C['dfl/'+k][source]['dev_clean']['seed17']['decisions'][DESIGN]['model_calls_mean'],k)) if valid else 'P0'
        policy=copy.deepcopy(values[ID]);family='dfl';best=family_key(base);family_table={}
        for fam in ('v3only','trace05'):
            cand=C[fam+'/P0'][source];reasons=clean_gate(base['dev_clean'],cand['dev_clean']);key=family_key(cand)
            # Better false-negative rate is mandatory; location/FP break remaining ties.
            if key[0]>=best[0]-1e-12:reasons.append('no actual-missing8 improvement on paired M1/M2/M2v2')
            family_table[fam]={'reasons':reasons,'key':key,'admissible':not reasons}
            if not reasons and key<best:family,best=fam,key
        profiles[target]={'source_selection_profile':source,'family':family,'policy':policy,
                          'adaptivity':'cross-profile internal development; already exposed E1-E9 and metadata; no independent qualification',
                          'provisional_combination':family!='dfl' and ID!='P0'}
        tables[target]={'policies':table,'families':family_table}
    original={p:copy.deepcopy(v['policy']) for p,v in profiles.items()}
    choices={v['policy']['id'] for v in profiles.values()}
    if len(choices)>1:
        def captured(ID):
            total=captured=0
            for prof in A.PROFILES:
                for c in A.SCREEN_CONDS:
                    for seed in seeds(C['dfl/'+ID][prof][c]):
                        m=seed['decisions'][DESIGN];total+=m['failures'];captured+=m['failures']-m['accepted_failures']
            return captured/total if total else 0.
        ID=max(sorted(choices),key=lambda k:(captured(k),-sum(cost(C['dfl/'+k][p]) for p in A.PROFILES)))
        for prof,item in profiles.items():
            item['policy']=copy.deepcopy(values[ID]);item['conflict_resolution']='larger observed failure capture among the two source-selected values'
            item['independently_selected_direction']=original[prof]['id']==ID
            item['provisional_combination']=item['family']!='dfl' and ID!='P0'
            item['adaptivity']+='; conflict resolution also uses both profile outcomes'
    else:
        for item in profiles.values():item['independently_selected_direction']=True
    return {'profiles':profiles,'selection_tables':tables,'evaluation79_GT_reads':0,
            'original_source_selected_policies':original,
            'same_policy_across_profiles':profiles[A.PROFILES[0]]['policy']==profiles[A.PROFILES[1]]['policy'],
            'formal_qualification':'UNQUALIFIED','policy_selection_requires_final_validation':True}
