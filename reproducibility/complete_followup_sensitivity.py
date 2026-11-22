"""Outcome-independent 45-day follow-up sensitivity; aggregate output only.

Requires authorized frozen episode records from recalculate_main_analysis.py and the two
source surveillance files. Original verdicts and all prior analyses are retained.
No row identifiers or source observations are written to the public output.
"""
from pathlib import Path
import argparse, hashlib, json
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--episodes',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'tables')
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    inputs=[a.episodes]; obs=[]
    for crop,mapping in [('apple',{'갈색무늬병(피해잎율)':'AMB','탄저병(피해과율)':'ABR'}),('rice',{'잎도열병(발생면적율)':'RB_IR'})]:
        src=a.project/'data/ncpms_api'/f'{crop}_surveillance_sigungu.csv'; inputs.append(src)
        t=pd.read_csv(src,dtype={'sidoCode':str,'inputStdrDatetm':str})
        t=t[t.dbyhsNm.isin(mapping)].copy();t['model']=t.dbyhsNm.map(mapping)
        t['county']=t.sigunguNm if crop=='apple' else t.sidoCode+'|'+t.sigunguNm
        t['date']=pd.to_datetime(t.inputStdrDatetm,format='%Y%m%d');t['year']=t.date.dt.year
        t=t[pd.to_numeric(t.inqireValue,errors='coerce').notna()]
        obs.append(t[['model','county','year','date']])
    last=pd.concat(obs).groupby(['model','county','year']).date.max().rename('last_survey').reset_index()
    ep=pd.read_csv(a.episodes,dtype={'county':str})
    assert ep.after_onset.dtype==bool
    ep=ep.merge(last,on=['model','county','year'],how='left',validate='many_to_one')
    ep['complete45']=ep.last_survey>=pd.to_datetime(ep.start)+pd.Timedelta(days=45)
    assert ep.loc[ep.verdict.eq('false_alarm'),'complete45'].all()
    rows=[]
    for key,base in ep.groupby(['model','threshold','comparison','support'],dropna=False):
        for phase in ['all','pre_or_no_onset']:
            d=base if phase=='all' else base[~base.after_onset]
            known=d[d.verdict.isin(['hit','false_alarm'])];full=known[known.complete45]
            rows.append(dict(zip(['model','threshold','comparison','support'],key))|{
                'phase':phase,'episodes_n':len(d),'ascertained_n':len(known),
                'false_n':int(known.verdict.eq('false_alarm').sum()),
                'ascertained_FAR':float(known.verdict.eq('false_alarm').mean()) if len(known) else None,
                'incomplete_positive_hits':int((known.verdict.eq('hit')&~known.complete45).sum()),
                'complete45_n':len(full),'complete45_false_n':int(full.verdict.eq('false_alarm').sum()),
                'complete45_FAR':float(full.verdict.eq('false_alarm').mean()) if len(full) else None})
    out=pd.DataFrame(rows).sort_values(['model','threshold','comparison','support','phase'])
    assert (out.false_n==out.complete45_false_n).all()
    assert (out.ascertained_n-out.complete45_n==out.incomplete_positive_hits).all()
    checks={('AMB',3):(118,499,390),('ABR',3):(48,114,101),('RB_IR',2):(2891,3882,3514),('RB_IR',4):(2607,3767,3427)}
    for (model,th),(f,n,n45) in checks.items():
        r=out[(out.model==model)&(out.threshold==th)&(out.comparison=='risk')&(out.support=='all')&(out.phase=='all')].iloc[0]
        assert (r.false_n,r.ascertained_n,r.complete45_n)==(f,n,n45)
    target=a.output/'DataS11_complete_followup.csv'
    if target.exists():pd.read_csv(target) # inspect prior content before replacement
    out.to_csv(target,index=False)
    record={'protocol':'Same requirement last same-year survey >= episode start + 45 days for hits and false alarms; retain original within-window observation/verdict criteria.',
      'interpretation':'Conditional ascertainment and complete-follow-up cohorts differ. Complete-follow-up excludes late-season episodes and does not ensure unbiased full-season FAR or absence of missed disease.',
      'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'aggregate_rows':len(out),
      'assertions':['all original false alarms have complete45 follow-up','false numerator unchanged in every stratum','denominator loss equals incomplete positive hits','four selected settings independently reproduced']}
    target=a.output/'DataS11_complete_followup_checks.json'
    if target.exists():json.loads(target.read_text(encoding='utf-8'))
    target.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(out[(out.comparison=='risk')&(out.support=='all')&(out.threshold.isin([2,3,4]))].to_string(index=False))

if __name__=='__main__':main()
