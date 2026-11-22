"""Main analysis: common support, paired controls, warning burden and observation sensitivity."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,datetime as dt,hashlib,json,platform,time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--rice-source',type=Path);a=ap.parse_args()
ROOT=a.project;OUT=Path(__file__).resolve().parents[1];SOURCES=[];SOURCE_NAMES={};START=time.time()
for n in ['tables','figures','results','logs']:(OUT/n).mkdir(parents=True,exist_ok=True)
def read(rel,**kw):
    p=a.rice_source if rel=='data/pairing/pairing_daily_rice.csv' and a.rice_source else ROOT/rel
    expected={'data/pairing/pairing_daily.csv':'7e436575d67fceab6f8c2884a3ca5168adef79fc479cbfd5079e5a32351fa207','data/pairing/pairing_daily_rice.csv':'dad24668e0cdd1696c0e2f7057bf4d55da8e83e0f8701664d292d9f072d86fbc','data/ncpms_api/apple_surveillance_sigungu.csv':'bc92a3689b43afd15cc8d699ec0f4599977720425f317754abeb311b09a63654','data/ncpms_api/rice_surveillance_sigungu.csv':'6ed10e859c6da3099ebb64fcf2c76109dda908749f017c125dbbb1c8a5506f78'}
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[rel],f'Input hash differs from raw-verified final source: {rel}'
    SOURCES.append(p);SOURCE_NAMES[p]=rel;return pd.read_csv(p,**kw)
def write(p,text):
    if p.exists():p.read_bytes()
    p.write_text(text,encoding='utf-8')
def csvfile(frame,name):write(OUT/'tables'/name,frame.to_csv(index=False))
def js(p,o):write(p,json.dumps(o,ensure_ascii=False,indent=2,default=lambda v:v.item()))
def ordinal(x):return dt.date.fromisoformat(str(x)).toordinal()
def iso(x):return dt.date.fromordinal(int(x)).isoformat()

ad=read('data/pairing/pairing_daily.csv');rd=read('data/pairing/pairing_daily_rice.csv',dtype={'sidoCode':str})
suopts={'dtype':{'inputStdrDatetm':str,'sidoCode':str,'sigunguCode':str}}
asu=read('data/ncpms_api/apple_surveillance_sigungu.csv',**suopts);rsu=read('data/ncpms_api/rice_surveillance_sigungu.csv',**suopts)
raw_counts={'apple':len(asu),'rice':len(rsu)}
ad['county']=ad.sigunguNm;rd['county']=rd.sidoCode+'|'+rd.sigunguNm
mapping={'갈색무늬병(피해잎율)':'AMB','탄저병(피해과율)':'ABR'}
asu=asu[asu.dbyhsNm.isin(mapping)].copy();asu['model']=asu.dbyhsNm.map(mapping);asu['county']=asu.sigunguNm
rsu=rsu[rsu.dbyhsNm=='잎도열병(발생면적율)'].copy();rsu['model']='RB_IR';rsu['county']=rsu.sidoCode+'|'+rsu.sigunguNm
d=pd.concat([ad,rd],ignore_index=True);d['year']=d.date.str[:4].astype(int);d=d[d.year<=2025].copy();d['day']=d.date.map(ordinal)
s=pd.concat([asu,rsu],ignore_index=True);s['year']=s.inputStdrDatetm.str[:4].astype(int);s['value']=pd.to_numeric(s.inqireValue,errors='coerce');s=s[(s.year<=2025)&s.value.notna()].copy();s['day']=s.inputStdrDatetm.map(lambda v:dt.datetime.strptime(v,'%Y%m%d').date().toordinal())
assert not d.duplicated(['model','county','day']).any()
assert not s.duplicated(['model','county','day']).any()
assert asu[asu.county.isin(ad.county.unique())].groupby('county').sigunguCode.nunique().max()==1
keys=set(map(tuple,d[['model','county']].drop_duplicates().to_numpy()));s=s[[tuple(v) in keys for v in s[['model','county']].to_numpy()]].copy()
province=s.groupby(['model','county']).sidoCode.first().to_dict()
survs={(m,c,y):(g.day.to_numpy(),g.value.to_numpy()) for (m,c,y),g in s.sort_values('day').groupby(['model','county','year'])}
RECORDS={};inventory=[]
zero={(m,int(y)) for (m,y),g in d.groupby(['model','year']) if (g.risk_max==0).all()}
for (m,c,y),g in d.sort_values('day').groupby(['model','county','year']):
    obs,vals=survs.get((m,c,y),(np.array([],dtype=int),np.array([])))
    RECORDS[m,c,int(y)]={'days':g.day.to_numpy(),'risk':g.risk_max.to_numpy(),'obs':obs,'values':vals,'province':province.get((m,c),'unavailable')}
for (m,y),g in d.groupby(['model','year']):
    z=s[(s.model==m)&(s.year==y)]
    inventory.append({'model':m,'year':int(y),'risk_counties':g.county.nunique(),'risk_days':len(g),'risk_nonzero_days':int((g.risk_max>0).sum()),'all_zero_risk':(m,int(y)) in zero,'survey_counties':z.county.nunique(),'survey_observations':len(z),'positive_surveys':int((z.value>0).sum())})
csvfile(pd.DataFrame(inventory),'DataS05_year_inventory.csv')

def evaluate(model,threshold,window=45,gap=7,support='all',calendar=None,doy_window=None):
    es=[];ps=[];bs=[]
    for (m,c,y),r in RECORDS.items():
        if m!=model:continue
        ds=r['days'];risk=r['risk'];obs=r['obs'];val=r['values']
        if support=='common':
            if not len(obs):continue
            mask=(ds>=obs.min()-45)&(ds<=obs.max());ds=ds[mask];risk=risk[mask]
        if not len(ds):continue
        if calendar:
            first,last=calendar;warn=(ds>=dt.date(y,*first).toordinal())&(ds<=dt.date(y,*last).toordinal())
        elif doy_window is not None:
            st,length=doy_window;doys=np.array([dt.date.fromordinal(int(x)).timetuple().tm_yday for x in ds]);warn=(doys>=st)&(doys<st+length)
        else:warn=risk>=threshold
        wd=ds[warn];pos=obs[val>0];on=int(pos.min()) if len(pos) else None
        bs.append({'model':m,'county':c,'year':y,'province':r['province'],'n_days':len(ds),'warning_days':int(warn.sum()),'surveys':len(obs)})
        if on is not None:
            pre=wd[(wd>=on-window)&(wd<=on)];same=wd[wd<=on];neg=obs[(obs<on)&(val==0)]
            lastneg=int(neg.max()) if len(neg) else None;hit=bool(len(pre));first=int(pre.min()) if hit else None
            signed=lastneg-first if lastneg is not None and first is not None else np.nan
            lower=signed if np.isfinite(signed) and signed>=0 else np.nan
            es.append({'model':m,'county':c,'year':y,'province':r['province'],'threshold':threshold,'onset':iso(on),'hit':hit,'all_zero_year':(m,y) in zero,'first_warning':iso(first) if first else '',
                'lead_window':on-first if hit else np.nan,'lead_season':on-int(same.min()) if hit and len(same) else np.nan,'lead_lower_nonnegative':lower,'lead_lower_signed':signed,
                'prior_negative':lastneg is not None,'negative_gap':on-lastneg if lastneg is not None else np.nan,'last_negative':iso(lastneg) if lastneg else ''})
        if len(wd):
            cuts=np.r_[0,np.where(np.diff(wd)>gap+1)[0]+1,len(wd)]
            for left,right in zip(cuts[:-1],cuts[1:]):
                st=int(wd[left]);en=int(wd[right-1]);inside=(obs>=st)&(obs<=st+window)
                if not len(obs) or st>obs.max():verdict='unverifiable'
                elif np.any((val>0)&inside):verdict='hit'
                elif obs.max()<st+window:verdict='censored'
                elif not np.any(inside):verdict='gap_unverifiable'
                else:verdict='false_alarm'
                ps.append({'model':m,'county':c,'year':y,'province':r['province'],'threshold':threshold,'start':iso(st),'end':iso(en),'warning_days':int(right-left),'verdict':verdict,'after_onset':on is not None and st>on,'onset':iso(on) if on else ''})
    return pd.DataFrame(es),pd.DataFrame(ps,columns=['model','county','year','province','threshold','start','end','warning_days','verdict','after_onset','onset']),pd.DataFrame(bs)

def percentile_counts(g,numerator,denominator,cluster):
    a=g.groupby(cluster)[[numerator,denominator]].sum()
    if len(a)<2 or a[denominator].sum()==0:return [np.nan,np.nan]
    rng=np.random.default_rng(20261026);ix=rng.integers(0,len(a),(5000,len(a)))
    den=a[denominator].to_numpy()[ix].sum(1);num=a[numerator].to_numpy()[ix].sum(1);v=num[den>0]/den[den>0]
    return np.quantile(v,[.025,.975]).tolist()
def summary(e,p,b,model,th,label,support,phase='all',intervals=True):
    q=p[~p.after_onset] if phase=='pre_or_no_onset' else p
    v=q[q.verdict.isin(['hit','false_alarm'])].copy();v['fa']=(v.verdict=='false_alarm').astype(int);v['one']=1
    et=e.copy();et['h']=et.hit.astype(int);et['one']=1;hit=et[et.hit];lo=hit.lead_lower_nonnegative.dropna()
    r={'model':model,'threshold':th,'comparison':label,'support':support,'episode_phase':phase,'events':len(et),'hits':int(et.hit.sum()),'POD':float(et.hit.mean()),'event_years':et.year.nunique(),'event_counties':et.county.nunique(),
       'episodes':len(q),'verifiable':len(v),'false_episodes':int(v.fa.sum()),'FAR':float(v.fa.mean()) if len(v) else np.nan,'censored':int((q.verdict=='censored').sum()),'unverifiable':int((q.verdict=='unverifiable').sum()),'gap_unverifiable':int((q.verdict=='gap_unverifiable').sum()),
       'post_onset_hit_episodes':int(((p.verdict=='hit')&p.after_onset).sum()),'lead_window_median':float(hit.lead_window.median()),'lead_season_median':float(hit.lead_season.median()),'lead_lower_median_conditional':float(lo.median()),'lower_nonnegative':len(lo),'lower_positive':int((lo>0).sum()),'lower_unconfirmed':len(hit)-len(lo),
       'common_or_all_days':int(b.n_days.sum()),'warning_days':int(b.warning_days.sum()),'warning_fraction':float(b.warning_days.sum()/b.n_days.sum())}
    if intervals:
        for cl in ['county','year']:
            for metric,g,num,den in [('POD',et,'h','one'),('FAR',v,'fa','one'),('warning_fraction',b,'warning_days','n_days')]:
                r[f'{metric}_{cl}_low'],r[f'{metric}_{cl}_high']=percentile_counts(g,num,den,cl)
    return r
def paired(er,ec,br,bc):
    q=er.merge(ec,on=['model','county','year','onset'],suffixes=('_risk','_calendar'),validate='one_to_one')
    assert len(q)==len(er)==len(ec)
    q['delta']=q.hit_risk.astype(int)-q.hit_calendar.astype(int);q['one']=1
    bd=br.merge(bc,on=['model','county','year'],suffixes=('_risk','_calendar'),validate='one_to_one')
    assert (bd.n_days_risk==bd.n_days_calendar).all()
    bd['delta_days']=bd.warning_days_risk-bd.warning_days_calendar
    r={'events':len(q),'risk_hits':int(q.hit_risk.sum()),'calendar_hits':int(q.hit_calendar.sum()),'risk_only':int((q.hit_risk&~q.hit_calendar).sum()),'calendar_only':int((~q.hit_risk&q.hit_calendar).sum()),'delta_POD':float(q.delta.mean()),'delta_warning_fraction':float(bd.delta_days.sum()/bd.n_days_risk.sum()),'support_days':int(bd.n_days_risk.sum()),'risk_warning_days':int(bd.warning_days_risk.sum()),'calendar_warning_days':int(bd.warning_days_calendar.sum())}
    for cl in ['county','year']:
        r[f'delta_POD_{cl}_low'],r[f'delta_POD_{cl}_high']=percentile_counts(q,'delta','one',cl)
        r[f'delta_warning_fraction_{cl}_low'],r[f'delta_warning_fraction_{cl}_high']=percentile_counts(bd,'delta_days','n_days_risk',cl)
    return r

SETTINGS=[('AMB',1),('AMB',2),('AMB',3),('ABR',1),('ABR',2),('ABR',3),('RB_IR',2),('RB_IR',4)]
CALENDARS={'July_August':((7,1),(8,31)),'June15_September15':((6,15),(9,15))}
CACHE={};allrows=[];allev=[];allep=[];allburden=[]
for support in ['all','common']:
    for model,t in SETTINGS:
        e,p,b=evaluate(model,t,support=support);CACHE[support,model,t,'risk']=(e,p,b)
        for phase in ['all','pre_or_no_onset']:allrows.append(summary(e,p,b,model,t,'risk',support,phase))
        for frame,coll in [(e,allev),(p,allep),(b,allburden)]:coll.append(frame.assign(comparison='risk',support=support,threshold=t))
    for model in ['AMB','ABR','RB_IR']:
        for label,calendar in CALENDARS.items():
            e,p,b=evaluate(model,0,support=support,calendar=calendar);CACHE[support,model,0,label]=(e,p,b)
            for phase in ['all','pre_or_no_onset']:allrows.append(summary(e,p,b,model,0,label,support,phase))
            for frame,coll in [(e,allev),(p,allep),(b,allburden)]:coll.append(frame.assign(comparison=label,support=support,threshold=0))
main=pd.DataFrame(allrows);csvfile(main,'DataS01_metrics_support_and_phase.csv')
# Explicit impact of the raw-supported Goseong correction; old counts are historical evidence, not constraints.
revision=[]
for model,t,h,n,f,v in [('AMB',3,102,155,118,499),('ABR',3,30,62,48,114),('RB_IR',2,662,669,2888,3879),('RB_IR',4,643,669,2606,3765)]:
    q=main[(main.model==model)&(main.threshold==t)&(main.comparison=='risk')&(main.support=='all')&(main.episode_phase=='all')].iloc[0]
    revision.append({'model':model,'threshold':t,'old_hits':h,'old_events':n,'old_false':f,'old_verifiable':v,'old_POD':h/n,'old_FAR':f/v,'raw_verified_hits':int(q.hits),'raw_verified_events':int(q.events),'raw_verified_false':int(q.false_episodes),'raw_verified_verifiable':int(q.verifiable),'raw_verified_POD':q.POD,'raw_verified_FAR':q.FAR})
csvfile(pd.DataFrame(revision),'DataS10_source_version_sensitivity.csv')
for name,frames in [('events',allev),('episodes',allep),('alarm_burden',allburden)]:write(OUT/'results'/f'{name}.csv',pd.concat(frames,ignore_index=True).to_csv(index=False))

pairrows=[]
for model,t in SETTINGS:
    e,p,b=CACHE['common',model,t,'risk']
    for label in CALENDARS:
        ce,cp,cb=CACHE['common',model,0,label]
        pairrows.append({'model':model,'threshold':t,'calendar':label,**paired(e,ce,b,cb)})
csvfile(pd.DataFrame(pairrows),'DataS02_paired_calendar_comparison.csv')

phase_null=[];replicates=[]
for model,t in SETTINGS:
    rng=np.random.default_rng(20261026);hits=np.zeros(100,dtype=int);events=0
    for (m,c,y),r in RECORDS.items():
        if m!=model or not len(r['obs']):continue
        obs=r['obs'];ds=r['days'];mask=(ds>=obs.min()-45)&(ds<=obs.max());ds=ds[mask];risk=r['risk'][mask];pos=obs[r['values']>0]
        if not len(ds) or not len(pos):continue
        assert np.all(np.diff(ds)==1),'Circular shift requires consecutive daily support'
        on=pos.min();eligible=(ds>=on-45)&(ds<=on);warn=risk>=t;events+=1
        for j,offset in enumerate(rng.integers(0,len(ds),100)):
            moved=np.roll(warn,int(offset));assert moved.sum()==warn.sum();hits[j]+=bool(moved[eligible].any())
    actual=CACHE['common',model,t,'risk'][0].hit.mean();values=hits/events
    phase_null.append({'model':model,'threshold':t,'events':events,'actual_POD':float(actual),'shift_POD_mean':float(values.mean()),'shift_POD_p025':float(np.quantile(values,.025)),'shift_POD_p975':float(np.quantile(values,.975)),'draws':100,'actual_percentile_among_shifts':float((values<=actual).mean())})
    replicates.extend({'model':model,'threshold':t,'replicate':j,'POD':float(v),'hits':int(hits[j]),'events':events} for j,v in enumerate(values))
csvfile(pd.DataFrame(phase_null),'DataS03_matched_warning_budget_shifts.csv');write(OUT/'results/phase_shift_replicates.csv',pd.DataFrame(replicates).to_csv(index=False))

seasonal=[]
for model,t in SETTINGS:
    e,p,b=CACHE['common',model,t,'risk'];train=e[e.year<=2019];bt=b[b.year<=2019]
    if len(train)==0 or len(bt)==0:
        seasonal.append({'model':model,'threshold':t,'status':'unavailable_no_early_positive_events'});continue
    length=int(round(bt.warning_days.mean()));best=(-1,None);eligible_doys=[]
    for r in train.itertuples():
        rec=RECORDS[model,r.county,int(r.year)];ds=rec['days'];obs=rec['obs'];on=ordinal(r.onset)
        kept=ds[(ds>=obs.min()-45)&(ds<=obs.max())&(ds>=on-45)&(ds<=on)]
        eligible_doys.append(np.array([dt.date.fromordinal(int(x)).timetuple().tm_yday for x in kept]))
    for start in range(1,367-max(1,length)):
        count=sum(bool(np.any((doys>=start)&(doys<start+length))) for doys in eligible_doys)
        if count>best[0]:best=(count,start)
    ce,cp,cb=evaluate(model,0,support='common',doy_window=(best[1],length))
    er=e[e.year>=2020];ec=ce[ce.year>=2020];br=b[b.year>=2020];bc=cb[cb.year>=2020]
    seasonal.append({'model':model,'threshold':t,'status':'fitted_early_tested_later','start_day_of_year':best[1],'window_length_days':length,'training_events':len(train),'training_calendar_hits':best[0],**paired(er,ec,br,bc)})
csvfile(pd.DataFrame(seasonal),'DataS04_temporal_seasonal_control.csv')

robust=[];observed=[];omissions=[]
for model,t in SETTINGS:
    e,p,b=CACHE['common',model,t,'risk'];ce,cp,cb=CACHE['common',model,0,'July_August']
    for label,years in [('through_2019',set(range(2014,2020))),('2020_2025',set(range(2020,2026)))]+[(f'year_{y}',{y}) for y in sorted(e.year.unique())]:
        et=e[e.year.isin(years)];pt=p[p.year.isin(years)];bt=b[b.year.isin(years)]
        if len(et):robust.append({'subset':label,**summary(et,pt,bt,model,t,'risk','common',intervals=True)})
    for tag,mask in [('all',np.ones(len(e),dtype=bool)),('prior_negative',e.prior_negative),('negative_gap_le15',e.negative_gap<=15),('negative_gap_le30',e.negative_gap<=30),('no_prior_negative',~e.prior_negative),('exclude_all_zero_archive_years',~e.all_zero_year)]:
        et=e[mask];ct=ce.merge(et[['county','year']],on=['county','year'],how='inner')
        if not len(et):continue
        q=et.assign(one=1,h=et.hit.astype(int));r={'model':model,'threshold':t,'tag':tag,'events':len(et),'hits':int(et.hit.sum()),'POD':float(et.hit.mean()),'calendar_hits':int(ct.hit.sum()),'event_years':et.year.nunique(),'event_counties':et.county.nunique()}
        for cl in ['county','year']:r[f'POD_{cl}_low'],r[f'POD_{cl}_high']=percentile_counts(q,'h','one',cl)
        observed.append(r)
    for prov in sorted(e.province.unique()):
        et=e[e.province!=prov];pt=p[p.province!=prov];bt=b[b.province!=prov]
        omissions.append({'omitted_province':str(prov),**summary(et,pt,bt,model,t,'risk','common',intervals=False)})
csvfile(pd.DataFrame(robust),'DataS06_temporal_sensitivity.csv');csvfile(pd.DataFrame(observed),'DataS07_observation_quality_sensitivity.csv');csvfile(pd.DataFrame(omissions),'DataS08_province_omission.csv')
sens=[]
for model,t in SETTINGS:
    for w in [30,45,60]:
        for gap in [3,7,14]:
            e,p,b=evaluate(model,t,w,gap,support='all');sens.append({'window':w,'merge_gap':gap,**summary(e,p,b,model,t,'risk','all',intervals=False)})
csvfile(pd.DataFrame(sens),'DataS09_window_gap_sensitivity.csv')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def savefig(name):
    plt.tight_layout()
    for ext in ['png','pdf']:
        p=OUT/'figures'/f'{name}.{ext}'
        if p.exists():p.read_bytes()
        plt.savefig(p,dpi=600,bbox_inches='tight')
    plt.close()
chosen=[('AMB',3),('ABR',3),('RB_IR',2),('RB_IR',4)];labels=['AMB >=3','ABR >=3','Rice >=2','Rice >=4']
fig,ax=plt.subplots(figsize=(7,4));x=np.arange(4)
for offset,phase,label in [(-.18,'all','All-season episodes'),(.18,'pre_or_no_onset','Starts before/on first positive, or no recorded positive')]:
    q=[main[(main.model==m)&(main.threshold==t)&(main.comparison=='risk')&(main.support=='all')&(main.episode_phase==phase)].iloc[0].FAR for m,t in chosen];ax.bar(x+offset,q,.36,label=label)
ax.set(xticks=x,xticklabels=labels,ylabel='Episode false-alarm ratio',ylim=(0,1));ax.legend(frameon=False,fontsize=8);savefig('figure1_episode_eligibility')
fig,axes=plt.subplots(1,2,figsize=(9,4))
for model,col in [('AMB','#237a8a'),('ABR','#d78e39'),('RB_IR','#9064a9')]:
    q=main[(main.model==model)&(main.comparison=='risk')&(main.support=='common')&(main.episode_phase=='all')]
    axes[0].plot(q.warning_fraction,q.POD,'o-',color=col,label=model)
    for r in q.itertuples():
        offset={('ABR',1):(4,9),('ABR',2):(-11,-1),('ABR',3):(4,-12)}.get((model,r.threshold),(4,4))
        axes[0].annotate(str(r.threshold),(r.warning_fraction,r.POD),xytext=offset,textcoords='offset points',fontsize=8)
    cal=main[(main.model==model)&(main.comparison=='July_August')&(main.support=='common')&(main.episode_phase=='all')].iloc[0];axes[0].scatter(cal.warning_fraction,cal.POD,marker='x',s=70,color=col)
axes[0].set(xlabel='Warning-day fraction on common support',ylabel='First-positive event POD',ylim=(0,1.07));axes[0].legend(frameon=False,fontsize=8)
pr=pd.DataFrame(pairrows);q=pr[(pr.calendar=='July_August')&(((pr.model.isin(['AMB','ABR']))&(pr.threshold==3))|((pr.model=='RB_IR')&(pr.threshold==2)))];yy=np.arange(len(q))
lo=np.minimum(q.delta_POD_county_low,q.delta_POD_year_low);hi=np.maximum(q.delta_POD_county_high,q.delta_POD_year_high)
axes[1].errorbar(q.delta_POD,yy,xerr=np.stack([q.delta_POD-lo,hi-q.delta_POD]),fmt='o',capsize=3);axes[1].axvline(0,color='gray',lw=1);axes[1].set(yticks=yy,yticklabels=[f'{r.model} >={r.threshold}' for r in q.itertuples()],xlabel='Risk minus July-August POD')
savefig('figure2_detection_burden_and_paired_difference')
fig,ax=plt.subplots(figsize=(7,4));null=pd.DataFrame(phase_null)
for j,(m,t) in enumerate(chosen):
    r=null[(null.model==m)&(null.threshold==t)].iloc[0];ax.errorbar(j-.1,r.shift_POD_mean,yerr=[[r.shift_POD_mean-r.shift_POD_p025],[r.shift_POD_p975-r.shift_POD_mean]],fmt='s',color='#888888',capsize=4,label='Circular-shift mean and central 95% range' if j==0 else None);ax.scatter(j+.1,r.actual_POD,color='#237a8a',label='Observed timing' if j==0 else None)
ax.set(xticks=np.arange(4),xticklabels=labels,ylabel='First-positive event POD',ylim=(0,1.07));ax.legend(frameon=False,fontsize=8);savefig('figure3_equal_budget_timing_control')
fig,ax=plt.subplots(figsize=(7,4));rq=pd.DataFrame(robust)
for m,t in [('AMB',3),('ABR',3),('RB_IR',2)]:
    q=rq[(rq.model==m)&(rq.threshold==t)&rq.subset.str.startswith('year_')].copy();ax.plot(q.subset.str[5:].astype(int),q.POD,'o-',label=f'{m} >={t}')
ax.set(xlabel='Year',ylabel='First-positive event POD',ylim=(0,1.07));ax.legend(frameon=False,fontsize=8);savefig('figure4_year_sensitivity')

meta={'raw_surveillance_rows':raw_counts,'analysis_daily_rows':len(d),'analysis_surveys':len(s),'all_zero_risk_years':sorted([[m,y] for m,y in zero]),'common_support':'same-year retained daily grid intersect [first valid survey minus45days,last valid survey]','phase_filter':'exclude only episode start strictly after first recorded positive; same-day retained; original verdict unchanged','bootstrap':{'replicates':5000,'seed':20261026,'separate_clusters':['county','year'],'joint_guarantee':False},'matched_budget_shift_draws':100,'historical_versions_verified':False,'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'elapsed_seconds':time.time()-START}
js(OUT/'results/run_metadata.json',meta);js(OUT/'reproducibility/source_checksums.json',[{'path':SOURCE_NAMES[p],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(SOURCES))]);js(OUT/'results/qa_summary.json',{'raw_verified_final_input_hashes':'passed','historical_counts':'source_revision_impact_table_preserved','matched_event_keys':'passed','paired_support_day_counts':'passed','shift_warning_budget_preserved':'passed','raw_inputs_modified':False})
print(main[['model','threshold','comparison','support','episode_phase','events','hits','POD','false_episodes','verifiable','FAR','warning_fraction']].round(4).to_string(index=False));print(json.dumps(meta,ensure_ascii=True,indent=2))
