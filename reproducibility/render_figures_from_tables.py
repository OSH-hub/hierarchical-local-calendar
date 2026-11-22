"""Recreate publication figures from the distributed aggregate CSV files only.
Does not load raw NCPMS records or modify manuscript text.
The CSV code RB_IR denotes rice leaf blast (RLB in the article).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]
main=pd.read_csv(OUT/'tables/DataS01_metrics_support_and_phase.csv')
pairrows=pd.read_csv(OUT/'tables/DataS02_paired_calendar_comparison.csv').to_dict('records')
phase_null=pd.read_csv(OUT/'tables/DataS03_matched_warning_budget_shifts.csv').to_dict('records')
robust=pd.read_csv(OUT/'tables/DataS06_temporal_sensitivity.csv').to_dict('records')
NAME={'AMB':'AMB','ABR':'ABR','RB_IR':'RLB'}
POD_LABEL='First-positive event POD'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def savefig(name):
    plt.tight_layout()
    for ext in ['png','pdf']:
        plt.savefig(OUT/'figures'/f'{name}.{ext}',dpi=600,bbox_inches='tight')
    plt.close()
chosen=[('AMB',3),('ABR',3),('RB_IR',2),('RB_IR',4)];labels=[f'{NAME[m]} ≥ {t}' for m,t in chosen]
fig,ax=plt.subplots(figsize=(7,4));x=np.arange(4)
for offset,phase,label in [(-.18,'all','All-season episodes'),(.18,'pre_or_no_onset','Starts before/on first positive, or no recorded positive')]:
    q=[main[(main.model==m)&(main.threshold==t)&(main.comparison=='risk')&(main.support=='all')&(main.episode_phase==phase)].iloc[0].FAR for m,t in chosen];ax.bar(x+offset,q,.36,label=label)
ax.set(xticks=x,xticklabels=labels,ylabel='Episode false-alarm ratio',ylim=(0,1));ax.legend(frameon=False,fontsize=8);savefig('figure1_episode_eligibility')
fig,axes=plt.subplots(1,2,figsize=(9,4))
# Label offsets (points) chosen so that threshold numbers do not overlap markers or connecting lines.
OFFSETS={('AMB',1):(5,-11),('AMB',2):(5,-11),('AMB',3):(5,-11),('ABR',1):(6,6),('ABR',2):(-12,-2),('ABR',3):(6,-12),('RB_IR',2):(6,-12),('RB_IR',4):(6,-12)}
for model,col in [('AMB','#237a8a'),('ABR','#d78e39'),('RB_IR','#9064a9')]:
    q=main[(main.model==model)&(main.comparison=='risk')&(main.support=='common')&(main.episode_phase=='all')]
    axes[0].plot(q.warning_fraction,q.POD,'o-',color=col,label=NAME[model])
    for r in q.itertuples():
        axes[0].annotate(str(r.threshold),(r.warning_fraction,r.POD),xytext=OFFSETS[(model,r.threshold)],textcoords='offset points',fontsize=8)
    cal=main[(main.model==model)&(main.comparison=='July_August')&(main.support=='common')&(main.episode_phase=='all')].iloc[0];axes[0].scatter(cal.warning_fraction,cal.POD,marker='x',s=70,color=col)
axes[0].set(xlabel='Warning-day fraction on common support',ylabel=POD_LABEL,ylim=(0,1.07));axes[0].legend(frameon=False,fontsize=8,loc='lower right')
axes[0].set_title('(a)',loc='left',fontsize=10,fontweight='bold')
pr=pd.DataFrame(pairrows);q=pr[(pr.calendar=='July_August')&(((pr.model.isin(['AMB','ABR']))&(pr.threshold==3))|((pr.model=='RB_IR')&(pr.threshold==2)))];yy=np.arange(len(q))
lo=np.minimum(q.delta_POD_county_low,q.delta_POD_year_low);hi=np.maximum(q.delta_POD_county_high,q.delta_POD_year_high)
axes[1].errorbar(q.delta_POD,yy,xerr=np.stack([q.delta_POD-lo,hi-q.delta_POD]),fmt='o',capsize=3);axes[1].axvline(0,color='gray',lw=1);axes[1].set(yticks=yy,yticklabels=[f'{NAME[r.model]} ≥ {r.threshold}' for r in q.itertuples()],xlabel='Risk minus July–August POD')
axes[1].set_title('(b)',loc='left',fontsize=10,fontweight='bold')
savefig('figure2_detection_burden_and_paired_difference')
fig,ax=plt.subplots(figsize=(7,4));null=pd.DataFrame(phase_null)
for j,(m,t) in enumerate(chosen):
    r=null[(null.model==m)&(null.threshold==t)].iloc[0];ax.errorbar(j-.1,r.shift_POD_mean,yerr=[[r.shift_POD_mean-r.shift_POD_p025],[r.shift_POD_p975-r.shift_POD_mean]],fmt='s',color='#888888',capsize=4,label='Circular-shift mean and central 95% range' if j==0 else None);ax.scatter(j+.1,r.actual_POD,color='#237a8a',label='Observed timing' if j==0 else None)
ax.set(xticks=np.arange(4),xticklabels=labels,ylabel=POD_LABEL,ylim=(0,1.07));ax.legend(frameon=False,fontsize=8,loc='lower left');savefig('figure3_equal_budget_timing_control')
fig,ax=plt.subplots(figsize=(7,4));rq=pd.DataFrame(robust)
for m,t in [('AMB',3),('ABR',3),('RB_IR',2)]:
    q=rq[(rq.model==m)&(rq.threshold==t)&rq.subset.str.startswith('year_')].copy();ax.plot(q.subset.str[5:].astype(int),q.POD,'o-',label=f'{NAME[m]} ≥ {t}')
ax.set(xlabel='Year',ylabel=POD_LABEL,ylim=(-0.03,1.07));ax.legend(frameon=False,fontsize=8);savefig('figure4_year_sensitivity')
