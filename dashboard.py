"""Defense-edition research dashboard for the existing disk anomaly pipeline.

This is a presentation layer. Detection is executed by dashboard_engine.py,
which imports the project's original algorithm modules from src/.
"""
from __future__ import annotations

import html
import json
import base64
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_engine import ROOT, load_run, parse_upload, run_dataset
from experiment_tables import TABLE_DESCRIPTIONS, generate_experiment_tables, load_table_csv
from table_4_6 import build_comparison
from table_4_8 import append_experiment_d, compare_c_d, parse_override_audit, verify_audit_against_metrics, Table48Error
from table_4_7 import parse_table47, parse_benign_stress, Table47Error

ASSETS = ROOT / "assets"
THESIS = ASSETS / "Thesis_Research_Document.pdf"
COLORS = {
    "surface": "#11223A",
    "muted": "#8EA6C0",
    "text": "#EDF4FC",
    "teal": "#52D6BC",
    "gold": "#D8B475",
    "red": "#D6869E",
    "blue": "#75B9EE",
    "purple": "#AB9CE8",
}
LOGO_FILE = ASSETS / "plm-logo.png"
LOGO_DATA = "data:image/png;base64," + base64.b64encode(LOGO_FILE.read_bytes()).decode("ascii")
TITLE = "Enhancement of Threshold-Based Algorithm for Detecting Abnormal Disk Usage Patterns Using System Logs"

st.set_page_config(
    page_title="Adaptive Disk Monitor | PLM Thesis Defense",
    page_icon=str(ASSETS / "favicon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "A research dashboard for the BSCS thesis, Pamantasan ng Lungsod ng Maynila."},
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--navy:#071626;--sidebar:#081b30;--panel:#10253e;--line:#294361;--text:#eef4fd;--muted:#a0b4cc;--blue:#4caff1;--gold:#dfb356;--red:#ef5c73;--green:#40ca9a}
*{box-sizing:border-box}
.stApp,[data-testid="stAppViewContainer"]{background:#071626;color:var(--text);font-family:'Inter','Segoe UI',Arial,sans-serif}
[data-testid="stHeader"]{background:rgba(7,22,38,.85)}
[data-testid="stMainBlockContainer"]{max-width:1540px;padding:1.5rem 2.15rem 3.5rem}
[data-testid="stSidebar"]{background:var(--sidebar);border-right:1px solid #29405e}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.48rem}
[data-testid="stSidebar"] div[data-testid="stButton"] button{width:100%;justify-content:flex-start;min-height:38px;border:none;background:transparent;color:#afc2d8;border-radius:8px;font-size:13px;font-weight:500;padding:.55rem .85rem;box-shadow:none}
[data-testid="stSidebar"] div[data-testid="stButton"] button:hover{background:#17324f;border:0;color:#fff}
[data-testid="stSidebar"] div[data-testid="stButton"] button:focus{outline:1px solid #386f9b;box-shadow:none}
[data-testid="stSidebar"] div[data-testid="stButton"].current-page button{background:#183d62;color:#f5faff;border-left:3px solid #65b6f0;border-radius:8px;font-weight:700}
[data-testid="stSidebar"] .st-key-selected-nav button{background:#183d62 !important;color:#f5faff !important;border-left:3px solid #65b6f0 !important}
[data-testid="stSidebar"] .stSlider label,[data-testid="stSidebar"] .stToggle label{color:#b5c9df}
[data-testid="stSidebar"] hr{border-color:#2b445d;margin:12px 0}
[data-testid="stMetric"]{background:var(--panel);border:1px solid #2b4662;border-radius:12px;padding:17px}
[data-testid="stMetricLabel"]{color:#a3bad1}[data-testid="stMetricValue"]{color:#f3f8ff}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:10px;overflow:hidden}
[data-testid="stFileUploader"]{background:#10263e;border:1px dashed #486683;border-radius:12px;padding:.6rem}
[data-testid="stFileUploaderDropzone"]{background:#10263e;border:1px dashed #486683}
[data-testid="stExpander"]{border:1px solid #2a4662;background:#10263f;border-radius:10px}
[data-baseweb="select"]>div,[data-baseweb="input"]>div{background-color:#132b46;border-color:#365370}
.stButton>button,.stDownloadButton>button{border-radius:9px;min-height:40px;font-weight:600}
button[kind="primary"]{background:#277fba !important;color:#fff !important;border:1px solid #348dc5 !important;box-shadow:none !important}
.stDownloadButton>button{background:#15304b;border:1px solid #31516f;color:#eaf4ff}
hr{border-color:#294563 !important}a{color:#7cc6f4!important}
h1,h2,h3,h4{font-family:'Inter','Segoe UI',Arial,sans-serif;letter-spacing:-.035em;color:#f3f7fd}
h2{font-size:25px}h3{font-size:18px}
small,[data-testid="stCaptionContainer"]{color:#a4bad1!important}
.brand{display:flex;align-items:center;gap:11px;padding:8px 4px 16px;margin:5px 0 6px;border-bottom:1px solid #24405a}
.brand-logo{width:58px;height:58px;object-fit:contain;flex-shrink:0}
.brand-name{font-size:14px;font-weight:800;color:#f3f7ff;letter-spacing:-.3px;line-height:1.35}
.brand-sub{font-size:10px;color:#9aafc7;line-height:1.5;margin-top:4px}
.side-label{font-size:10px;color:#7591b3;letter-spacing:1.5px;font-weight:750;text-transform:uppercase;margin:17px 8px 7px}
.side-footer{padding:16px 7px 3px;border-top:1px solid #28445f;color:#88a2be;font-size:10px;line-height:1.7}
.app-top{display:flex;justify-content:space-between;align-items:center;padding:0 0 16px;margin:0 0 17px;border-bottom:1px solid #203e5c}
.app-top strong{font-size:13px;color:#dbe8f6;font-weight:700}.app-top span{font-size:10px;color:#96adc3;letter-spacing:.25px}
.hero{background:linear-gradient(115deg,#112d4d,#173a59);border:1px solid #365a7d;border-radius:16px;padding:30px 35px 23px;margin-bottom:20px}
.hero-main{display:grid;grid-template-columns:112px minmax(0,1fr);align-items:center;gap:24px}
.hero-logo{height:108px;width:108px;object-fit:contain}
.hero-kicker{color:#e4bd77;font-size:10px;letter-spacing:2.2px;font-weight:750;text-transform:uppercase;margin-bottom:13px}
.hero-lines{height:2px;width:90px;background:linear-gradient(90deg,#db687e,#e7bd66,#3acba5);border-radius:3px;margin-bottom:14px}
.hero h1{font-size:clamp(28px,2.8vw,40px);line-height:1.25;margin:0 0 15px;font-weight:800;max-width:820px}
.hero p{color:#c0d2e5;font-size:12px;line-height:1.75;margin:0;max-width:910px}
.hero-foot{display:flex;flex-wrap:wrap;gap:12px 29px;border-top:1px solid #3a5d7b;margin-top:24px;padding-top:15px;color:#acc3da;font-size:11px;line-height:1.6}
.hero-foot strong{color:#f6f9ff;margin-right:5px}
.section-header{margin:10px 0 18px}
.section-kicker{color:#e1b672;font-size:10px;font-weight:750;letter-spacing:1.7px;text-transform:uppercase;margin:0 0 8px}
.section-header h2{font-size:23px;margin:0 0 7px;font-weight:750}
.section-header p{font-size:12px;color:#a4bad2;line-height:1.65;max-width:880px;margin:0}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin:17px 0 21px}
.stat-card{background:#102741;border:1px solid #2c4c6b;border-radius:12px;padding:17px 18px 18px;min-height:114px}
.stat-card:before{content:'';display:block;background:var(--stat-color,#4caff1);height:2px;width:38px;margin-bottom:14px;border-radius:4px}
.stat-label{font-size:10px;color:#a4b9d0;letter-spacing:.8px;font-weight:700;text-transform:uppercase}
.stat-value{font-size:27px;letter-spacing:-.8px;font-weight:800;color:#f2f7ff;margin:7px 0 5px}
.stat-foot{color:#91a9c3;font-size:11px;line-height:1.4}
.panel{background:#10263e;border:1px solid #2b4764;border-radius:12px;padding:18px 19px;margin:5px 0 15px}
.panel-title{font-weight:700;font-size:15px;color:#eef5ff;margin:1px 0 5px}
.panel-caption{color:#9aafc7;font-size:11px;margin-bottom:12px}
.research-label{display:inline-block;font-size:10px;color:#ecc58b;font-weight:700;letter-spacing:1px;text-transform:uppercase;border:1px solid #6b5b4e;border-radius:6px;background:#2b2b32;padding:5px 10px;margin-bottom:8px}
.info-strip{background:#112a44;border:1px solid #31536f;border-radius:11px;color:#bcd0e3;padding:15px 18px;margin:12px 0 20px;font-size:12px;line-height:1.75}
.empty{background:#112a44;border:1px dashed #3c607c;border-radius:11px;color:#a6bed5;padding:25px;margin:9px 0 20px;font-size:12px;line-height:1.7}
.empty strong{display:block;color:#e8f3ff;font-size:17px;margin:0 0 6px}
.card{background:#112a44;border:1px solid #2d4e6d;border-radius:11px;padding:17px;margin:7px 0 15px}
.card h3{font-size:15px;margin:0 0 8px}.card p{font-size:12px;color:#b1c7dc;line-height:1.7}
.person{background:#112940;border:1px solid #304e6a;border-radius:10px;padding:15px 16px;margin:6px 0;min-height:88px}
.person-role{font-size:10px;color:#d8b779;font-weight:700;letter-spacing:.8px;text-transform:uppercase;margin-bottom:9px}
.person-name{font-size:13px;color:#ecf4fc;font-weight:650;line-height:1.5}
.paper-title{font-size:23px;line-height:1.45;font-weight:750;color:#f3f8ff;margin:14px 0}
.method-stage{border-left:3px solid #58b9e9;background:#112a44;padding:14px;border-radius:4px 10px 10px 4px;min-height:90px;margin:8px 0}
.method-stage b{display:block;color:#e9f4ff;font-size:12px;margin-bottom:6px}.method-stage span{display:block;color:#acc1d6;font-size:11px;line-height:1.55}
.run-pill{display:inline-block;border:1px solid #2b7161;border-radius:6px;background:#102e2d;color:#67ddb3;font-size:10px;font-weight:700;letter-spacing:.4px;padding:5px 9px}
.alert-card{display:flex;align-items:flex-start;justify-content:space-between;gap:13px;padding:14px 15px;margin-bottom:8px;background:#122c45;border:1px solid #2c4d6b;border-radius:10px}
.alert-priority{font-size:9px;letter-spacing:1px;font-weight:700;text-transform:uppercase;margin-bottom:5px}
.alert-name{font-size:13px;font-weight:700;color:#f4f8fe}.alert-meta{font-size:10px;color:#a4bbd2;line-height:1.7;margin-top:4px}
.alert-score{font-weight:800;font-size:19px;color:#e9f3fc}
@media(max-width:1100px){.hero{padding:24px}.hero-main{grid-template-columns:84px 1fr;gap:18px}.hero-logo{height:84px;width:84px}.stats{grid-template-columns:repeat(2,minmax(0,1fr));}}
@media(max-width:700px){[data-testid="stMainBlockContainer"]{padding:1.1rem}.hero-main{display:block}.hero-logo{height:65px;width:65px;margin-bottom:12px}.hero h1{font-size:28px}.hero-foot{gap:8px 18px}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}.stat-value{font-size:21px}}
</style>
""", unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value))


def section(code, title, subtitle):
    st.markdown(f'<div class="section-header"><div class="section-kicker">{esc(code)}</div>'
                f'<h2>{esc(title)}</h2><p>{esc(subtitle)}</p></div>', unsafe_allow_html=True)


def tiles(items):
    html_cards = ''.join(
        '<div class="stat-card" style="--stat-color:%s"><div class="stat-label">%s</div>'
        '<div class="stat-value">%s</div><div class="stat-foot">%s</div></div>'
        % (color, esc(label), esc(value), esc(note))
        for label, value, note, color in items
    )
    st.markdown(f'<div class="stats">{html_cards}</div>', unsafe_allow_html=True)


def empty(message, detail):
    st.markdown(f'<div class="empty"><strong>{esc(message)}</strong>{esc(detail)}</div>',unsafe_allow_html=True)


def person(name, role):
    st.markdown(f'<div class="person"><div class="person-role">{esc(role)}</div>'
                f'<div class="person-name">{esc(name)}</div></div>',unsafe_allow_html=True)


def chart_style(fig, height=330, show_legend=True):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, Segoe UI, sans-serif',color='#A9BDD1',size=11),
        margin=dict(t=18,b=25,l=35,r=20), height=height,
        hovermode='x unified', legend=dict(orientation='h',y=1.16,x=0,font=dict(size=11)),
        showlegend=show_legend,
    )
    fig.update_xaxes(showgrid=False,zeroline=False,showline=True,linecolor='#344B61',tickfont=dict(color='#93A9BF'))
    fig.update_yaxes(showgrid=True,gridcolor='rgba(124,149,171,.14)',zeroline=False,tickfont=dict(color='#93A9BF'))
    return fig


def comparison_chart(data, cols=('Precision','Recall','F1')):
    color_map={'Precision':'#52D6BC','Recall':'#75B9EE','F1':'#D8B475'}
    fig=go.Figure()
    for metric in cols:
        fig.add_bar(x=data['Method'],y=data[metric]*100,name=metric,marker_color=color_map[metric],
                    text=[f'{v:.1%}' for v in data[metric]],textposition='outside',textfont=dict(size=10,color='#C8D9E7'))
    fig.update_layout(barmode='group', bargap=.28,yaxis=dict(title='Score (%)',range=[0,110]))
    chart_style(fig,360)
    st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})


# Navigation with real buttons; selections persist across Streamlit reruns.
def go_to_page(target):
    st.session_state['page'] = target

if 'page' not in st.session_state:
    st.session_state['page'] = 'Overview'
page = st.session_state['page']
st.sidebar.markdown(
    f'<div class="brand"><img class="brand-logo" src="{LOGO_DATA}" alt="PLM seal">'
    '<div><div class="brand-name">Pamantasan ng<br>Lungsod ng Maynila</div>'
    '<div class="brand-sub">BS Computer Science<br>Thesis Defense Dashboard</div></div></div>',
    unsafe_allow_html=True,
)
nav_groups = [
    ('WORKSPACE', ['Overview', 'Thesis Information', 'Dataset Laboratory']),
    ('MONITORING', ['Monitoring Overview', 'Alert Intelligence']),
    ('RESEARCH', ['Methodology', 'Experiment Tables', 'Research Evaluation']),
    ('SYSTEM', ['Settings']),
]
for group_title, links in nav_groups:
    st.sidebar.markdown(f'<div class="side-label">{group_title}</div>', unsafe_allow_html=True)
    for item in links:
        st.sidebar.button(item, key=('selected-nav' if page == item else 'nav-' + item.lower().replace(' ', '-')),
                          on_click=go_to_page, args=(item,), use_container_width=True)

# Keep experiment configuration available in the same workspace for repeatable testing.
with st.sidebar.expander('Detection parameters', expanded=page in ('Dataset Laboratory', 'Settings')):
    window=st.slider('Rolling window (W)',10,500,50,10)
    min_obs=st.slider('Warm-up observations',3,min(window,100),min(10,window))
    theta=st.slider('Z-score threshold',0.5,10.0,5.5,0.5)
    use_ceiling=st.toggle('Hybrid safety ceiling',True)
    ceiling=st.slider('Disk ceiling (%)',50,100,85) if use_ceiling else None
    debounce=st.slider('Consecutive readings',1,6,1)
    delta=st.slider('Correlation window (seconds)',0,300,30,5)
    suppress_benign=st.toggle('Suppress benign-context alerts',True,
        help='Enabled: documented enhanced mode. Disabled: priority-only hybrid ablation.')
    protect_critical_alerts=st.toggle('Protect critical-ceiling alerts (Experiment D)',False,
        help='When suppression is ON, never suppress a disk reading at or above the active safety ceiling. OFF reproduces A/B/C.')
    srgg_enabled=st.toggle('Experimental SRGG detection',False,
        help='Additional seasonal-growth detection; requires regular 5-minute readings and 1–3 days history.')
    srgg_reference_days=st.slider('SRGG historical days',1,3,3)
    srgg_persistence=st.slider('SRGG persistence (readings)',1,6,3)
    srgg_diff=st.slider('SRGG seasonal difference (pp)',3.,20.,11.,1.)
    srgg_growth=st.slider('SRGG hourly growth (pp)',1.,8.,3.,0.5)
    srgg_floor=st.slider('SRGG utilization floor (%)',30.,80.,55.,1.)

folder_str=st.session_state.get('run_folder','')
folder=Path(folder_str) if folder_str else None
summary=rows=df=None
if folder and (folder/'summary.json').exists():
    try:
        summary,rows=load_run(folder)
        df=pd.DataFrame(rows)
        if not df.empty:
            df['timestamp']=pd.to_datetime(df['timestamp'],errors='coerce')
            df['suppressed']=df['suppressed'].fillna(False).astype(bool)
            df['actionable']=df['classification'].eq('ABNORMAL') & ~df['suppressed']
    except (ValueError,KeyError,OSError,json.JSONDecodeError) as exc:
        st.warning(f'Saved run could not be loaded: {exc}')
        folder=summary=rows=df=None

st.sidebar.markdown('<div class="side-label">CURRENT SESSION</div>', unsafe_allow_html=True)
if summary:
    st.sidebar.markdown('<span class="run-pill">LOADED EXPERIMENT</span>', unsafe_allow_html=True)
    st.sidebar.caption(f"{summary['run_id']} | {summary['readings']:,} readings")
else:
    st.sidebar.caption('No experiment loaded yet. Start in Dataset Laboratory.')
st.sidebar.markdown('<div class="side-footer">College of Information Systems and Technology Management<br>'
                    'Computer Science Department<br>Research prototype | 2026</div>',unsafe_allow_html=True)

st.markdown('<div class="app-top"><strong>Disk Anomaly Monitoring System</strong>'
            '<span>PLM · BS COMPUTER SCIENCE · RESEARCH PROTOTYPE</span></div>',unsafe_allow_html=True)
if page == 'Overview':
    st.markdown(f'''<div class="hero"><div class="hero-main"><img class="hero-logo" alt="Pamantasan ng Lungsod ng Maynila seal" src="{LOGO_DATA}"/>
    <div><div class="hero-kicker">Pamantasan ng Lungsod ng Maynila / BS Computer Science</div><div class="hero-lines"></div>
    <h1>Intelligent disk usage.<br>Evidence-driven alerts.</h1>
    <p>{esc(TITLE)}.</p></div></div>
    <div class="hero-foot"><span><strong>Researchers</strong> Khurt Luis L. Co &amp; Patrick Jerome D. Bacani</span>
    <span><strong>Research Adviser</strong> Prof. Ariel Antwaun Rolando C. Sison</span>
    <span><strong>Academic Year</strong> 2026</span></div></div>''',unsafe_allow_html=True)

if page in ('Overview', 'Monitoring Overview'):
    section('MONITORING / ANALYSIS',
            'Monitoring overview' if page == 'Monitoring Overview' else 'Dashboard overview',
            'Results from the selected dataset and the research pipeline. Experiment values are never simulated.')
    if summary and df is not None and not df.empty:
        latest=df.sort_values('timestamp').iloc[-1]
        current_usage=f'{float(latest["disk_usage_value"]):.1f}%'
        actionable=df[df['actionable']]
        if summary.get('metrics') and summary['metrics']['matched']:
            m=summary['metrics']
            precision=f'{m["precision"]:.1%}'
            recall=f'{m["recall"]:.1%}'
            performance_note=f'{m["matched"]:,} matched labels'
        else:
            precision=recall='—'
            performance_note='Upload independent labels to calculate'
        tiles([
            ('Latest disk reading',current_usage,'Last uploaded disk observation','#50b3ee'),
            ('Actionable alerts',f'{len(actionable):,}',f'{summary["readings"]:,} readings processed','#ef6378'),
            ('Precision',precision,performance_note,'#e2b64d'),
            ('Recall',recall,performance_note,'#46cd9e'),
        ])
        st.caption(f'Loaded experiment: {summary["run_id"]}  |  Source: {", ".join(summary["uploaded_files"])}  |  Historical file replay (not live telemetry)')
        left,right=st.columns([1.8,1],gap='medium')
        with left:
            st.markdown('<div class="panel-title">Disk usage trend</div><div class="panel-caption">Actual utilization and sliding-window baseline</div>',unsafe_allow_html=True)
            sample=df.dropna(subset=['timestamp']).sort_values('timestamp')
            if len(sample)>2200:
                sample=sample.iloc[::max(1,len(sample)//2000)]
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=sample['timestamp'],y=sample['disk_usage_value'],name='Disk usage',
                            mode='lines',line=dict(color='#ed6179',width=2),fill='tozeroy',
                            fillcolor='rgba(237,97,121,.10)'))
            if 'baseline_mu' in sample:
                fig.add_trace(go.Scatter(x=sample['timestamp'],y=sample['baseline_mu'],name='Dynamic baseline',
                                mode='lines',line=dict(color='#e6b54d',width=2,dash='dash')))
            markers=sample[sample['actionable']]
            if not markers.empty:
                fig.add_trace(go.Scatter(x=markers['timestamp'],y=markers['disk_usage_value'],name='Detected alerts',
                                mode='markers',marker=dict(color='#f26d81',size=8,symbol='circle-open',line=dict(width=2))))
            fig.update_yaxes(title='Disk usage (%)',range=[0,102])
            chart_style(fig,355)
            st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
        with right:
            st.markdown('<div class="panel-title">Alert priority distribution</div><div class="panel-caption">Delivered alerts after contextual policy</div>',unsafe_allow_html=True)
            counts=actionable.priority.value_counts().reindex(['HIGH','MEDIUM','LOW'],fill_value=0)
            if counts.sum():
                fig=go.Figure(go.Pie(labels=['High','Medium','Low'],values=counts.values,hole=.67,
                                    marker=dict(colors=['#ed6179','#e9b74e','#46c99b']),sort=False,
                                    textinfo='none'))
                fig.update_layout(annotations=[dict(text=f'<b>{counts.sum():,}</b><br>alerts',showarrow=False,font=dict(size=17,color='#eef6ff'))])
                chart_style(fig,310,True)
                fig.update_layout(legend=dict(orientation='v',y=.75,x=1.0),margin=dict(t=15,b=15,l=10,r=55))
                st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
            else:
                empty('No actionable alerts','The selected detection policy produced no delivered alerts.')
        st.markdown('### Recent / top alerts')
        if not actionable.empty:
            top=actionable.sort_values('enhanced_alert_score',ascending=False).head(6)
            columns=['timestamp','disk_usage_value','zscore','priority','enhanced_alert_score']
            columns=[c for c in columns if c in top.columns]
            st.dataframe(top[columns].rename(columns={'timestamp':'Timestamp','disk_usage_value':'Disk usage (%)','zscore':'Z-score','priority':'Priority','enhanced_alert_score':'Alert score'}),
                         hide_index=True,use_container_width=True)
        else:
            st.info('No alerts to display. Adjust detection parameters or upload a different dataset.')
        st.caption('All displayed readings, alerts and metrics originate from the actual Python detection pipeline.')
    else:
        tiles([
            ('Latest disk reading','—','Run a dataset to display','#50b3ee'),
            ('Actionable alerts','—','Not yet analyzed','#ed6179'),
            ('Precision','—','Requires labeled data','#e6b64e'),
            ('Recall','—','Requires labeled data','#46ca9a'),
        ])
        left,right=st.columns([1.8,1],gap='medium')
        with left:
            st.markdown('### Disk usage trend')
            empty('No dataset has been processed',
                  'Open Dataset Laboratory to run the example logs or upload a CSV, LOG or TXT dataset. Usage trends, baselines and anomaly markers will appear here.')
            if st.button('Go to Dataset Laboratory',type='primary'):
                go_to_page('Dataset Laboratory')
                st.rerun()
        with right:
            st.markdown('### Alert priority distribution')
            empty('Awaiting detector output','Priorities will appear after contextual correlation and alert scoring.')
        st.info('No results are displayed until you upload a dataset and execute an experiment. Thesis manuscript values are not inserted as test results.')

elif page=='Dataset Laboratory':
    section('02 / Experimental workspace','Dataset laboratory',
            'Run the original pipeline on your own logs or the provided research samples. Every experiment is stored separately.')
    with st.container(border=True):
        st.markdown('### 1. Select a data source')
        choice=st.radio('Source',['Upload custom dataset','Example CSV dataset','Included full Linux log'],horizontal=True)
        datasets=[]
        if choice=='Upload custom dataset':
            ups=st.file_uploader('Upload LOG, TXT or disk usage CSV files',type=['log','txt','csv'],accept_multiple_files=True,
                help='Choose your own prepared dataset. For CSV use timestamp,disk_usage_value and optional message columns.')
            if ups:
                for f in ups:
                    try:
                        source=f.getvalue()
                        text_lines=parse_upload(f.name,source)
                        datasets.append((f.name,source))
                        st.success(f'{f.name} · {len(text_lines.splitlines()):,} input lines accepted')
                    except ValueError as exc: st.error(f'{f.name}: {exc}')
        elif choice=='Example CSV dataset':
            f=ROOT/'data/examples/demo_disk_usage.csv'
            if f.exists():
                datasets=[(f.name,f.read_bytes())]
                st.success(f'Example selected · {f.name} · {len(f.read_text().splitlines())-1} rows')
                st.dataframe(pd.read_csv(f).head(8),hide_index=True,use_container_width=True)
            else: st.error('Example dataset not found.')
        elif choice=='Included full Linux log':
            f=ROOT/'data/raw/Linux_full.log'
            if f.exists():
                st.info(f'Full benchmark input selected: {f.name} ({f.stat().st_size/(1024*1024):.1f} MB).')
                datasets=[(f.name,f.read_bytes())]
            else: st.error('Included Linux log file not found.')
        st.divider()
        st.markdown('### 2. Optional ground-truth evaluation')
        demo_labels=False
        if choice=='Example CSV dataset' and (ROOT/'data/examples/demo_labels.csv').exists():
            demo_labels=st.checkbox('Use accompanying example labels (demonstration only)',value=True)
        labels_file=st.file_uploader('Upload independently labeled observations',type=['csv','jsonl'],
            help='Optional. Must contain timestamp and is_anomaly. This is required for precision, recall and F1.')
        labels_data=labels_file.getvalue() if labels_file else None
        labels_name=labels_file.name if labels_file else ''
        label_provenance=st.selectbox('How were the reference labels created?',[
            'Independently reviewed labels', 'Synthetic injection labels',
            'Rule/threshold-derived labels', 'Not specified'],index=1 if demo_labels else 3,
            help='This is recorded in the experiment manifest. Synthetic or threshold-derived labels do not establish real-world generalization.')
        if not labels_data and demo_labels:
            lab=ROOT/'data/examples/demo_labels.csv'
            labels_data=lab.read_bytes()
            labels_name=lab.name
        st.caption('Ground truth should be established independently of the detection rules. Without labels, the dashboard displays counts and anomaly scores but not accuracy metrics.')
        st.divider()
        st.markdown('### 3. Execute and save experiment')
        st.caption('Change detection parameters using Algorithm settings in the left sidebar. All raw uploads and outputs are preserved under runs/.')
        if st.button('Run detection analysis',type='primary',use_container_width=True,disabled=not datasets):
            try:
                with st.spinner('Processing logs through parsing, baseline calculation, classification, correlation and scoring...'):
                    run_folder,result=run_dataset(datasets,labels_data,labels_name,window,min_obs,theta,ceiling,debounce,delta,suppress_benign,label_provenance,
                        srgg_enabled=srgg_enabled, srgg_reference_days=srgg_reference_days,
                        srgg_persistence=srgg_persistence, srgg_diff=srgg_diff,
                        srgg_growth=srgg_growth, srgg_floor=srgg_floor,
                        protect_critical_alerts=protect_critical_alerts)
                    st.session_state['run_folder']=str(run_folder)
                st.success(f'Experiment completed · {result["readings"]:,} disk observations · {result["alerts"]:,} delivered alerts. Open Experiment Tables to run the five chapter-style tests.')
            except Exception as exc:
                st.error(f'Pipeline execution failed: {exc}')
    section('Saved work','Experiment history','Open an earlier run without overwriting your original thesis benchmark or the current configuration.')
    saved=sorted((p for p in (ROOT/'runs').glob('*') if (p/'summary.json').exists()),reverse=True)
    if saved:
        selected=st.selectbox('Choose experiment',saved,format_func=lambda p:p.name)
        if st.button('Load selected experiment'):
            st.session_state['run_folder']=str(selected)
            st.rerun()
    else: st.caption('No saved experiments found.')
    with st.expander('Input format and validation rules'):
        st.code('timestamp,disk_usage_value,message\n2024-03-01 09:00:00,42,normal operation\n2024-03-01 09:05:00,93,I/O error',language='csv')
        st.code('timestamp,is_anomaly\n2024-03-01T09:00:00,false\n2024-03-01T09:05:00,true',language='csv')
        st.markdown('For `.log` or `.txt`, use timestamped Linux system-log lines containing a disk usage reading such as `93% used`. Uploaded CSV files are converted into syslog-style lines for the original parser. A usable dataset requires enough unique disk readings for baseline warm-up.')
        f=ROOT/'data/examples/demo_disk_usage.csv'
        if f.exists():st.download_button('Download example dataset',f.read_bytes(),file_name='demo_disk_usage.csv',mime='text/csv')
        f=ROOT/'data/examples/demo_labels.csv'
        if f.exists():st.download_button('Download example labels',f.read_bytes(),file_name='demo_labels.csv',mime='text/csv')

elif page=='Alert Intelligence':
    section('03 / Detection output','Alert intelligence',
            'Inspect ranked disk anomalies, audit decisions, context keywords and abnormality scores.')
    if df is None or df.empty:
        empty('No alert stream available','Run a dataset in the laboratory to generate an actual ranked alert stream.')
    else:
        abnormal=df.loc[df['classification'].eq('ABNORMAL')]
        actionable=df.loc[df.actionable]
        tiles([
            ('Flagged readings',f'{len(abnormal):,}','Before contextual suppression','#D6869E'),
            ('Delivered alerts',f'{len(actionable):,}','Actionable output','#52D6BC'),
            ('High priority',f'{(actionable.priority=="HIGH").sum():,}','Failure evidence','#D8B475'),
            ('Suppressed',f'{int(df.suppressed.sum()):,}','Retained in audit history','#75B9EE'),
        ])
        st.markdown('### High-Priority Alert Occurrences')
        high_alerts = actionable.loc[actionable['priority'].eq('HIGH')].copy()

        if high_alerts.empty:
            st.info('No delivered HIGH-priority alerts in this experiment.')
        else:
            # These are matched log keywords, not confirmed root causes.
            if 'matched_keyword' not in high_alerts.columns:
                high_alerts['matched_keyword'] = None
            high_alerts['context_label'] = (
                high_alerts['matched_keyword']
                .fillna('No matched log context')
                .astype(str)
                .str.strip()
                .replace('', 'No matched log context')
            )

            frequency = (
                high_alerts.groupby('context_label', as_index=False)
                .agg(
                    occurrences=('context_label', 'size'),
                    highest_utilization=('disk_usage_value', 'max'),
                )
                .sort_values(['occurrences', 'context_label'], ascending=[False, True])
            )

            fig = go.Figure(
                go.Bar(
                    x=frequency['occurrences'],
                    y=frequency['context_label'],
                    orientation='h',
                    marker=dict(color='#36C5EF', line=dict(color='#85DDF9', width=0.5)),
                    text=frequency['occurrences'].map(lambda n: f'{n:,}'),
                    textposition='outside',
                    cliponaxis=False,
                    customdata=frequency[['highest_utilization']].to_numpy(),
                    hovertemplate=(
                        '<b>%{y}</b><br>'
                        'High-priority alert readings: %{x}<br>'
                        'Highest observed disk usage: %{customdata[0]:.1f}%'
                        '<extra></extra>'
                    ),
                )
            )
            fig.update_layout(
                title=dict(text='Alert Count by Matched Log Context', font=dict(size=17)),
                template='plotly_dark',
                paper_bgcolor='#102741',
                plot_bgcolor='#102741',
                font=dict(color='#EDF4FC', family='Inter, Arial, sans-serif'),
                showlegend=False,
                margin=dict(l=12, r=58, t=60, b=55),
                height=max(315, 165 + len(frequency) * 55),
                bargap=0.28,
                xaxis=dict(
                    title='Number of high-priority alert readings',
                    rangemode='tozero',
                    range=[0, max(1, frequency['occurrences'].max()) * 1.22 + 0.3],
                    gridcolor='#2B4662',
                    zeroline=False,
                ),
                yaxis=dict(title='', autorange='reversed', showgrid=False),
            )
            if frequency['occurrences'].max() <= 15:
                fig.update_xaxes(dtick=1)

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.caption(
                f'{len(high_alerts):,} delivered high-priority alert readings across '
                f'{len(frequency)} matched log-context categories. '
                'Counts are readings, not unique incidents. '
                'Matched keywords are supporting context, not confirmed causes.'
            )
        st.markdown('### Detailed event explorer')
        cat=st.selectbox('Classification',['Delivered alerts','All anomalous events','All readings'])
        view=actionable if cat=='Delivered alerts' else abnormal if cat=='All anomalous events' else df
        filter_priorities=st.multiselect('Priority filter',['HIGH','MEDIUM','LOW'],default=['HIGH','MEDIUM','LOW'])
        if cat!='All readings':view=view.loc[view.priority.isin(filter_priorities)]
        view=view.sort_values('enhanced_alert_score',ascending=False)
        columns=['timestamp','log_source','disk_usage_value','baseline_mu','zscore','classification','triggered_by','srgg_eligible','srgg_reference','srgg_seasonal_delta','srgg_hourly_growth','srgg_streak','srgg_new_detection','priority','matched_keyword','enhanced_alert_score','critical_override_applied','suppressed']
        columns=[col for col in columns if col in view.columns]
        st.dataframe(view[columns],use_container_width=True,hide_index=True,height=425)
        c1,c2,c3=st.columns(3)
        c1.download_button('Export filtered alerts',view[columns].to_csv(index=False).encode('utf-8'),file_name='filtered_alerts.csv',mime='text/csv',use_container_width=True)
        scored_file=folder/'final_scored_stream.jsonl'
        c2.download_button('Export scored stream',scored_file.read_bytes(),file_name='final_scored_stream.jsonl',mime='application/x-ndjson',use_container_width=True)
        audit_file=folder/'audit_trail.jsonl'
        c3.download_button('Export audit trail',audit_file.read_bytes(),file_name='audit_trail.jsonl',mime='application/x-ndjson',use_container_width=True)
        st.caption('Suppressed readings remain available in the audit trail. Enhanced alert scores are assigned by the original algorithm, not generated for display.')

elif page=='Research Evaluation':
    section('04 / Your experiment','Algorithm performance evaluation',
        'Only the selected experiment is evaluated here. Accuracy metrics require uploaded ground-truth labels.')
    if summary and summary.get('metrics') is not None:
        m=summary['metrics']
        st.markdown(f'<span class="run-pill">RUN {esc(summary["run_id"])}</span>',unsafe_allow_html=True)
        st.caption(f'Source: {", ".join(summary["uploaded_files"])}  |  Labels: {summary.get("labels_filename") or "None"}  |  Provenance: {summary.get("label_provenance", "Not specified")}')
        if m['matched']==0:
            st.error('No uploaded label timestamps matched baseline-ready observations. Metrics cannot be interpreted.')
        else:
            tiles([
                ('Precision',f'{m["precision"]:.1%}',f'{m["tp"]} true positives','#52D6BC'),
                ('Recall',f'{m["recall"]:.1%}',f'{m["fn"]} false negatives','#75B9EE'),
                ('F1 score',f'{m["f1"]:.1%}','From current experiment','#D8B475'),
                ('False-positive rate',f'{m["fpr"]:.3%}',f'{m["fp"]} false positives','#D6869E'),
            ])
            c1,c2=st.columns([1,1],gap='large')
            with c1:
                st.markdown('#### Confusion matrix')
                z=[[m['tp'],m['fn']],[m['fp'],m['tn']]]
                fig=go.Figure(go.Heatmap(z=z,x=['Predicted anomaly','Predicted normal'],y=['Actual anomaly','Actual normal'],
                    text=z,texttemplate='%{text:,}',colorscale=[[0,'#153349'],[1,'#42BDAA']],showscale=False))
                chart_style(fig,290,False);st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
            with c2:
                st.markdown('#### Static vs. enhanced (current settings)')
                variants=[('Enhanced (current)',m),('Static 80%',summary.get('static_80_metrics')),('Static 85%',summary.get('static_85_metrics'))]
                vals=[{'Method':name,'Precision':mm['precision'],'Recall':mm['recall'],'F1':mm['f1']} for name,mm in variants if mm]
                if vals:comparison_chart(pd.DataFrame(vals))
            st.caption(f'{m["matched"]:,} timestamps matched out of {m["labels_count"]:,} labels; the warm-up interval is excluded from scoring.')
            if m['matched']<m['labels_count']:
                st.warning('Some labels did not match eligible readings. Check timestamps and baseline warm-up before interpreting metrics.')
            if 'Synthetic' in summary.get('label_provenance','') or 'Rule/threshold' in summary.get('label_provenance',''):
                st.warning('These labels are not independently verified production incidents. Report synthetic/threshold-label limitations during your defense.')
        st.download_button('Export this experiment’s metrics',json.dumps(m,indent=2),file_name='current_evaluation.json',mime='application/json')
        st.markdown('### Generate Chapter 4-style tables from your uploaded dataset')
        if st.button('Open Experiment Tables',type='primary',key='go_exp_tables_eval'):
            go_to_page('Experiment Tables');st.rerun()
    elif summary:
        empty('Labels required for evaluation','Upload your own labeled observations in Dataset Laboratory, then run the detector again. No results will be invented.')
    else:
        empty('No experiment has been run','Go to Dataset Laboratory, upload your data and execute the detector first.')
        if st.button('Go to Dataset Laboratory',type='primary',key='eval_to_lab'):
            go_to_page('Dataset Laboratory');st.rerun()

elif page=='Experiment Tables':
    section('RESEARCH / USER-RUN TESTING','Experiment results tables',
        'Generate Chapter 4-style Tables 4.1–4.5 using your uploaded data and labels. The manuscript figures are never prefilled into these tables.')
    if not summary:
        empty('No uploaded experiment available','Open Dataset Laboratory to upload your dataset, labels, and run the detection first.')
        if st.button('Go to Dataset Laboratory',key='tables_to_lab',type='primary'):
            go_to_page('Dataset Laboratory');st.rerun()
    elif not (folder/'ground_truth_uploaded').exists():
        st.warning('This run does not contain stored uploaded ground-truth labels. Re-run Dataset Laboratory with a label file to generate performance tables.')
        if st.button('Upload labeled dataset',key='tables_to_labels'):
            go_to_page('Dataset Laboratory');st.rerun()
    else:
        st.caption(f'Active experiment: {summary["run_id"]}  |  Source: {", ".join(summary["uploaded_files"])}  |  Labels: {summary.get("labels_filename")}')
        st.info('The tables below will be calculated only when you click Generate selected tables. Parameter sweeps re-run the original detection modules on your data. Larger logs and grids require more computation.')
        with st.expander('Configure testing parameters',expanded=True):
            c1,c2,c3=st.columns(3)
            with c1:
                w_tables=st.number_input('Rolling window W',min_value=10,max_value=1000,value=50,step=10,key='table_window')
                theta_tables=st.number_input('Final Z-score threshold',min_value=0.5,max_value=20.0,value=5.5,step=0.5,key='table_theta')
            with c2:
                min_tables=st.number_input('Warm-up observations',min_value=3,max_value=int(w_tables),value=min(10,int(w_tables)),key='table_min')
                ceiling_tables=st.number_input('Final safety ceiling (%)',min_value=1.0,max_value=100.0,value=85.0,step=1.0,key='table_ceil')
            with c3:
                debounce_tables=st.number_input('Consecutive readings',min_value=1,max_value=10,value=1,key='table_debounce')
                delta_tables=st.number_input('Correlation ±seconds',min_value=0,max_value=3600,value=30,step=5,key='table_delta')
            table_selection=st.multiselect('Tables to calculate',list(TABLE_DESCRIPTIONS),default=list(TABLE_DESCRIPTIONS),
                format_func=lambda code:f'Table {code} — {TABLE_DESCRIPTIONS[code]}',key='table_selection')
            st.caption('Table 4.2 uses a per-window warm-up of max(5, 20% of W), matching the project’s original window-sweep script.')
            grid_mode=st.checkbox('Use full 63-combination grid for Table 4.3 (instead of selected thesis-style comparisons)',value=False,key='full_grid')
            resamples=st.select_slider('Bootstrap resamples for Table 4.5',options=[200,1000,5000,10000],value=5000,key='bootstrap_count')
        if st.button('Generate selected tables from this experiment',type='primary',disabled=not table_selection,use_container_width=True,key='run_table_generation'):
            status=st.empty()
            try:
                with st.spinner('Running parameter sweeps and computing evaluation tables on your dataset...'):
                    target,manifest=generate_experiment_tables(folder,selected=table_selection,
                         window=int(w_tables),min_obs=int(min_tables),theta_final=float(theta_tables),
                         ceiling_final=float(ceiling_tables),delta=int(delta_tables),
                         debounce=int(debounce_tables),joint_full=bool(grid_mode),
                         bootstrap_resamples=int(resamples),
                         callback=lambda msg:status.caption(msg))
                st.success(f'Completed and saved {len(manifest["tables"])} tables for experiment {summary["run_id"]}.')
            except Exception as exc:
                st.error(f'Table generation failed: {exc}')
        table_root=folder/'experiment_tables'
        batches=sorted((p for p in table_root.glob('*') if p.is_dir() and (p/'manifest.json').exists()),reverse=True)
        table_dir=st.selectbox('View generated table batch',batches,format_func=lambda p:p.name,key='select_generated_batch') if batches else table_root
        manifest_path=table_dir/'manifest.json'
        if batches and manifest_path.exists():
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
            st.divider()
            st.markdown('### Generated results — your own test data')
            st.caption(f'Computed from {manifest["source_run_id"]} | {manifest["matched_input_timestamps"]:,} matched timestamps | Label origin: {manifest["label_provenance"]}')
            for table_id in TABLE_DESCRIPTIONS:
                csv_path=table_dir/f'table_{table_id.replace(".","_")}.csv'
                if not csv_path.exists() or table_id not in manifest['tables']:continue
                st.markdown(f'#### Table {table_id} — {TABLE_DESCRIPTIONS[table_id]}')
                data=load_table_csv(table_dir,table_id)
                st.dataframe(data,use_container_width=True,hide_index=True,height=min(540,90+35*len(data)))
                st.download_button(f'Download Table {table_id} (CSV)',csv_path.read_bytes(),
                    file_name=f'{manifest["source_run_id"]}_table_{table_id.replace(".","_")}.csv',
                    mime='text/csv',key=f'download_experiment_{table_id}')
                if table_id=='4.5' and (table_dir/'bootstrap_detail.json').exists():
                    details=json.loads((table_dir/'bootstrap_detail.json').read_text())
                    low,high=details['f1_difference_95_ci']
                    st.caption(f'Paired Enhanced − Static 80% F1 difference: {details["f1_difference_mean"]:+.4f}; 95% CI [{low:+.4f}, {high:+.4f}]; {details["resamples"]:,} resamples.')
            st.download_button('Download experiment manifest',manifest_path.read_bytes(),
                file_name=f'{summary["run_id"]}_table_manifest.json',mime='application/json')
            if 'Synthetic' in manifest['label_provenance'] or 'Rule/threshold' in manifest['label_provenance']:
                st.warning('These results use non-independent labels; they are suitable for controlled testing but not proof of generalization to real-world incidents.')
        else:
            empty('No tables generated yet','Configure and run the tests above. No manuscript figures are inserted into the output.')


    # TABLE_4_6_INCREMENTAL_COMPARISON_START
    st.divider()
    st.markdown('### Table 4.6 — Incremental Algorithm Performance Comparison')
    st.caption('Compare three previously exported Research Evaluation metrics files. This is independent of the original Tables 4.1–4.5 sensitivity sweeps.')
    with st.expander('Upload A / B / C metrics for Table 4.6', expanded=True):
        ca, cb, cc = st.columns(3)
        with ca:
            table46_a = st.file_uploader('A — Existing detector', type=['json'], key='table46_a')
        with cb:
            table46_b = st.file_uploader('B — Existing + Original SRGG', type=['json'], key='table46_b')
        with cc:
            table46_c = st.file_uploader('C — Full pipeline with suppression', type=['json'], key='table46_c')
        table46_d = st.file_uploader('D — SRGG + critical-ceiling override (optional)', type=['json'], key='table46_d')
        st.warning('The metrics exports contain no dataset ID or parameter settings. Confirm A, B, C use the same dataset, synthetic injection labels and scoring settings; only SRGG/suppression should differ. Matching counts do not guarantee matching data.')
        if all([table46_a, table46_b, table46_c]):
            try:
                m46_a = json.loads(table46_a.getvalue().decode('utf-8-sig'))
                m46_b = json.loads(table46_b.getvalue().decode('utf-8-sig'))
                m46_c = json.loads(table46_c.getvalue().decode('utf-8-sig'))
                table46_rows = build_comparison(m46_a, m46_b, m46_c)
                if table46_d is not None:
                    m46_d = json.loads(table46_d.getvalue().decode('utf-8-sig'))
                    table46_rows = append_experiment_d(table46_rows, m46_d)
            except (ValueError, UnicodeError) as exc:
                st.error(f'Cannot generate Table 4.6: {exc}')
            else:
                table46_frame = pd.DataFrame(table46_rows)
                st.markdown('#### Table 4.6 — Results from your uploaded experiments')
                st.dataframe(table46_frame, use_container_width=True, hide_index=True)
                plot46 = pd.DataFrame([
                    {'Method': f'Experiment {row["Experiment"]}',
                     'Precision': row['Precision (%)'] / 100,
                     'Recall': row['Recall (%)'] / 100,
                     'F1': row['F1-score (%)'] / 100}
                    for row in table46_rows
                ])
                st.markdown('#### Precision, Recall, and F1 by experiment')
                comparison_chart(plot46)
                gain_ab = table46_rows[1]['F1-score (%)'] - table46_rows[0]['F1-score (%)']
                gain_bc = table46_rows[2]['F1-score (%)'] - table46_rows[1]['F1-score (%)']
                c1, c2 = st.columns(2)
                c1.metric('F1 change: B versus A', f'{gain_ab:+.2f} percentage points')
                c2.metric('F1 change: C versus B', f'{gain_bc:+.2f} percentage points')
                if len(table46_rows) == 4:
                    gain_cd = table46_rows[3]['F1-score (%)'] - table46_rows[2]['F1-score (%)']
                    st.metric('F1 change: D versus C', f'{gain_cd:+.2f} percentage points')
                st.download_button('Download Table 4.6 (CSV)',
                    table46_frame.to_csv(index=False).encode('utf-8'),
                    file_name='table_4_6_incremental_algorithm_performance.csv',
                    mime='text/csv', key='download_table46')
                st.caption('Interpretation: additional SRGG detections are not independently verified here. These exports report classification metrics on the synthetic injection-label benchmark; event-level early warning and benign-growth tests are separate requirements.')
        else:
            st.info('Upload all three JSON files to calculate Table 4.6. Files may be named A_metrics.json, B_metrics.json, C_metrics.json, or C_metrics(1).json.')
    # TABLE_4_6_INCREMENTAL_COMPARISON_END

    # TABLE_4_7_VERIFIED_EPISODE_COMPARISON_START
    st.divider()
    st.markdown('### Table 4.7 — Gradual-Degradation Detection Evaluation')
    st.caption('Import a label-verified episode CSV derived from the aligned A/B/C scored streams and the original injected labels. This table does not alter detection decisions.')
    t47_upload = st.file_uploader('Upload Table 4.7 label-verified episode CSV', type=['csv'], key='table47_upload')
    if t47_upload is None:
        st.info('Upload Table_4_7_Label_Verified_Gradual_Degradation.csv, included in the installation kit. Rebuild this CSV if you change the dataset, labels, or algorithm settings.')
    else:
        try:
            t47_rows, t47_stats = parse_table47(t47_upload.getvalue())
        except (Table47Error, UnicodeError, ValueError) as exc:
            st.error(f'Cannot load Table 4.7: {exc}')
        else:
            t47_df = pd.DataFrame(t47_rows)
            st.dataframe(t47_df, use_container_width=True, hide_index=True)
            x1,x2,x3 = st.columns(3)
            x1.metric('Mean existing delay', f'{t47_stats["mean_existing_delay_min"]:.1f} min')
            x2.metric('Mean SRGG delay', f'{t47_stats["mean_srgg_delay_min"]:.1f} min')
            x3.metric('Mean improvement', f'{t47_stats["mean_improvement_min"]:+.1f} min')
            t47_chart = go.Figure()
            for col,label,color in [
                ('Existing delay (min)','Existing detector','#75B9EE'),
                ('SRGG delay (min)','Original SRGG','#52D6BC')]:
                t47_chart.add_trace(go.Bar(name=label,x=t47_df['Episode'],y=t47_df[col],marker_color=color))
            t47_chart.update_layout(barmode='group',height=330,
                yaxis_title='Minutes after injected-label onset',
                margin=dict(l=10,r=10,t=20,b=30),
                paper_bgcolor='#10263e',plot_bgcolor='#10263e',
                font=dict(color='#EDF4FC'),legend=dict(orientation='h',y=1.13))
            st.plotly_chart(t47_chart,use_container_width=True,config={'displayModeBar':False})
            st.warning(f'The {t47_stats["total_preonset_false_positives"]} pre-onset SRGG alerts in these episodes are false positives under the supplied injection-label definition. Do not silently relabel them.')
            st.caption('Detection delay is measured from the first labeled positive to the first delivered alert within that labeled episode. A zero-minute delay is not proof of zero physical failure-detection latency. These results were derived from a synthetic benchmark.')
            st.download_button('Download Table 4.7 (CSV)',
                t47_df.to_csv(index=False).encode('utf-8'),
                file_name='table_4_7_verified_gradual_degradation.csv',
                mime='text/csv',key='download_table47')
    st.divider()
    st.markdown('### Supplementary Validation — Legitimate Growth Stress Test')
    st.caption('This validation uses deliberately normal synthetic workloads. Its purpose is to identify false-alert limitations of SRGG, not to inflate F1.')
    t47stress_upload = st.file_uploader('Upload benign_growth_stress_results.csv',type=['csv'],key='table47stress')
    if t47stress_upload is not None:
        try:
            t47stress_rows = parse_benign_stress(t47stress_upload.getvalue())
        except (Table47Error,UnicodeError,ValueError) as exc:
            st.error(f'Cannot load stress results: {exc}')
        else:
            t47stress_df=pd.DataFrame(t47stress_rows)
            st.dataframe(t47stress_df,use_container_width=True,hide_index=True)
            pivot=t47stress_df.pivot(index='Scenario',columns='Method',values='False-alert observations')
            plot47stress=go.Figure()
            for j,method in enumerate(pivot.columns):
                plot47stress.add_trace(go.Bar(name=str(method),x=pivot.index,
                                             y=pivot[method].fillna(0),
                                             marker_color=['#75B9EE','#52D6BC','#D8B475'][j%3]))
            plot47stress.update_layout(barmode='group',height=360,
                yaxis_title='False-alert observations (lower is better)',
                margin=dict(l=5,r=5,t=20,b=95),paper_bgcolor='#10263e',
                plot_bgcolor='#10263e',font=dict(color='#EDF4FC'))
            st.plotly_chart(plot47stress,use_container_width=True,config={'displayModeBar':False})
            st.warning('These are researcher-defined normal scenarios. Every alert counts as a false alert in this test. They do not establish general production performance.')
            st.download_button('Download stress comparison (CSV)',
                t47stress_df.to_csv(index=False).encode('utf-8'),
                file_name='srgg_benign_growth_validation.csv',
                mime='text/csv',key='download_table47_stress')
    else:
        st.info('Run validate_benign_growth.py from the installation kit, then upload runs/srgg_validation/benign_growth_stress_results.csv to visualize the detector tradeoff.')
    # TABLE_4_7_VERIFIED_EPISODE_COMPARISON_END

    # TABLE_4_8_CRITICAL_OVERRIDE_COMPARISON_START
    st.divider()
    st.markdown('### Table 4.8 — Contextual Suppression and Critical-Ceiling Override Evaluation')
    st.caption('Measure the difference between Experiment C (suppression without override) and Experiment D (suppression with critical-ceiling override). This is alert handling, not a second detection algorithm.')
    with st.expander('Upload Experiment C / D metrics and verified suppression audit', expanded=True):
        t48_left, t48_right = st.columns(2)
        with t48_left:
            t48_c = st.file_uploader('C — SRGG + suppression (current_evaluation.json)', type=['json'], key='table48_c')
        with t48_right:
            t48_d = st.file_uploader('D — SRGG + suppression + critical override (current_evaluation.json)', type=['json'], key='table48_d')
        t48_audit = st.file_uploader('Critical override audit: Experiment_D_critical_override_audit.csv', type=['csv'], key='table48_audit')
        st.caption('Use C_metrics.json, D_metrics.json, and the CSV produced by comparing C_scored and D_scored with the injection labels. Different filenames are fine; matching totals alone do not guarantee identical dataset provenance.')
        if t48_c is not None and t48_d is not None:
            try:
                t48_metrics_c = json.loads(t48_c.getvalue().decode('utf-8-sig'))
                t48_metrics_d = json.loads(t48_d.getvalue().decode('utf-8-sig'))
                t48_rows, t48_delta = compare_c_d(t48_metrics_c, t48_metrics_d)
            except (Table48Error, ValueError, UnicodeError) as exc:
                st.error(f'Table 4.8 metrics error: {exc}')
            else:
                t48_df = pd.DataFrame(t48_rows)
                st.dataframe(t48_df, use_container_width=True, hide_index=True)
                t48_stats1, t48_stats2, t48_stats3 = st.columns(3)
                t48_stats1.metric('Restored true-positive readings', f'{t48_delta["tp"]:+d}')
                t48_stats2.metric('Change in false positives', f'{t48_delta["fp"]:+d}')
                t48_stats3.metric('F1 change (D vs C)', f'{t48_delta["f1_pp"]:+.2f} pp')
                t48_chart=go.Figure()
                for col,label,color in [('Recall (%)','Recall','#75B9EE'),('F1-score (%)','F1-score','#52D6BC')]:
                    t48_chart.add_trace(go.Bar(name=label, x=t48_df['Experiment'], y=t48_df[col], marker_color=color))
                t48_chart.update_layout(barmode='group', height=320, yaxis_title='Percent', yaxis_range=[0, 105],
                    margin=dict(l=10,r=10,t=20,b=30), paper_bgcolor='#10263e', plot_bgcolor='#10263e',
                    font=dict(color='#EDF4FC'), legend=dict(orientation='h',y=1.13))
                st.plotly_chart(t48_chart, use_container_width=True, config={'displayModeBar':False})
                st.download_button('Download Table 4.8 comparison (CSV)',
                    t48_df.to_csv(index=False).encode('utf-8'),
                    file_name='table_4_8_critical_override_comparison.csv', mime='text/csv', key='download_table48')
                if t48_audit is not None:
                    try:
                        t48_evidence, t48_audit_summary = parse_override_audit(t48_audit.getvalue(), ceiling=85.0)
                        verify_audit_against_metrics(t48_delta, t48_audit_summary)
                    except (Table48Error, ValueError, UnicodeError) as exc:
                        st.error(f'Table 4.8 audit could not be reconciled with the metrics: {exc}')
                    else:
                        st.success(f'Verified {t48_audit_summary["restored"]} critical alert readings restored. Override audit and metric changes agree.')
                        st.markdown('#### Restored alerts by contextual keyword')
                        t48_kw_df=pd.DataFrame([{'Keyword':key,'Restored alerts':value} for key,value in sorted(t48_audit_summary['keywords'].items(),key=lambda p:-p[1])])
                        st.dataframe(t48_kw_df, use_container_width=True, hide_index=True)
                        st.markdown('#### Detailed restored alert records')
                        t48_evidence_df=pd.DataFrame(t48_evidence)
                        st.dataframe(t48_evidence_df,use_container_width=True,hide_index=True)
                        st.download_button('Download verified Table 4.8 restored-alert audit (CSV)',
                            t48_evidence_df.to_csv(index=False).encode('utf-8'),
                            file_name='table_4_8_verified_restored_alerts.csv', mime='text/csv', key='download_table48_audit')
                else:
                    st.info('Upload the optional override audit CSV to verify exactly which critical alert readings were restored.')
                st.warning('C to D measures a suppression safeguard, not a new detection enhancement. Report effects only for the synthetic injection-label benchmark; stronger real-world validation is still needed.')
        else:
            st.info('Upload the C and D metrics JSON files to calculate Table 4.8. The audit CSV provides observation-level verification.')
    # TABLE_4_8_CRITICAL_OVERRIDE_COMPARISON_END

elif page=='Thesis Information':
    section('05 / Institutional profile','Thesis and research team',
            'Official research details extracted from the supplied title page and approval sheet, with methodology and findings summarized from the manuscript.')
    c1,c2=st.columns([1.8,0.9],gap='large')
    with c1:
        st.markdown('<span class="research-label">Undergraduate thesis / November 2026</span>',unsafe_allow_html=True)
        st.markdown(f'<div class="paper-title">{esc(TITLE)}</div>',unsafe_allow_html=True)
        st.markdown('**Pamantasan ng Lungsod ng Maynila** — University of the City of Manila, Intramuros, Manila')
        st.caption('College of Information Systems and Technology Management · Computer Science Department · Bachelor of Science in Computer Science')
        st.markdown('### Researchers')
        a,b=st.columns(2)
        with a:person('Co, Khurt Luis L.','Researcher')
        with b:person('Bacani, Patrick Jerome D.','Researcher')
        st.markdown('### Academic guidance')
        a,b=st.columns(2)
        with a:person('Prof. Ariel Antwaun Rolando C. Sison','Thesis adviser')
        with b:person('Prof. Raymund M. Dioses','Research coordinator')
        st.markdown('### Panel of examiners')
        a,b=st.columns(2)
        with a:person('Prof. Raymund M. Dioses','Panel member')
        with b:person('Prof. Connie Adriano','Panel member')
        if THESIS.exists():
            st.download_button('Download source thesis manuscript (PDF)',THESIS.read_bytes(),
                file_name='PLM_Disk_Usage_Thesis_2026.pdf',mime='application/pdf',type='primary')
    with c2:
        st.markdown('### Research cover')
        cover=ASSETS/'thesis_cover.png'
        if cover.exists():st.image(str(cover),use_column_width=True,caption='Uploaded thesis manuscript, cover page')
    st.divider()
    st.markdown('### Research architecture')
    a,b,c,d=st.columns(4,gap='medium')
    for target,name,description in [(a,'01 / Log acquisition','Linux syslog, kernel and filesystem log input'),
                                    (b,'02 / Dynamic baseline','Sliding-window mean and standard deviation'),
                                    (c,'03 / Hybrid detection','Z-score comparison with critical safety ceiling'),
                                    (d,'04 / Correlation & ranking','Context-based severity and scored alert output')]:
        with target:
            st.markdown(f'<div class="method-stage"><b>{esc(name)}</b><span>{esc(description)}</span></div>',unsafe_allow_html=True)
    st.markdown('### Scope and research interpretation')
    st.markdown('''The study focuses on disk usage in Linux systems and uses a rule-based statistical detector rather than deep learning. The validation described in the thesis uses a controlled disk-usage series mixed with Linux system logs, not measurements from an operational production deployment.

The documented final evaluation distinguishes the **enhanced configuration with benign-context suppression** from a **priority-only hybrid configuration**. These should not be presented as interchangeable algorithms or as live-production validation.''')
    st.caption('Sources: manuscript title page, approval sheet, sections 1.5 and 3.2, and Chapter 4 (particularly Table 4.4). The title displayed here follows the manuscript cover page; the approval sheet uses a slightly different title wording.')

elif page=='Methodology':
    section('RESEARCH / METHODS','System methodology',
            'The processing stages of the enhanced threshold-based algorithm, as described in Chapter 3 of the thesis.')
    for num,label,body in [
        ('01','Log acquisition and preprocessing','Read Linux system logs, parse unstructured messages using Drain, and normalize disk usage observations.'),
        ('02','Dynamic baseline','Calculate moving mean and standard deviation using a sliding window, with baseline warm-up and dispersion handling.'),
        ('03','Anomaly scoring','Apply Z-score classification, optional disk-usage safety ceiling, and consecutive-reading debounce.'),
        ('04','Contextual correlation','Match related disk and filesystem events to explain detections and prioritize or suppress context-dependent alerts.'),
        ('05','Audit, ranking and evaluation','Produce ranked alerts, preserve the full audit trail, and calculate precision, recall and F1 only if independent ground-truth labels are supplied.'),
    ]:
        st.markdown(f'<div class="method-stage"><b>{num} / {esc(label)}</b><span>{esc(body)}</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="info-strip">This is a dataset-replay research prototype, not a continuously connected production server-monitoring service. Its documented validation uses a controlled, synthetic disk-usage benchmark combined with Linux system logs.</div>',unsafe_allow_html=True)

elif page=='Settings':
    section('SYSTEM / CONFIGURATION','Experiment settings',
            'Use Detection parameters in the sidebar to adjust the algorithm before each new run. Changes never overwrite an existing experiment.')
    tiles([
        ('Sliding window',str(window),'Prior observations','#50b3ee'),
        ('Z-score cutoff',f'{theta:.1f}','Detection sensitivity','#e2b64d'),
        ('Safety ceiling',str(ceiling)+'%' if ceiling is not None else 'Off','Absolute critical limit','#ed6179'),
        ('Correlation policy','Suppress' if suppress_benign else 'Priority only','Benign context decision','#46ca9a'),
    ])
    st.markdown('### Recommended demonstration flow')
    st.markdown('''1. Go to **Dataset Laboratory** and run the included example CSV with its labels.
2. Open **Overview** to present disk readings, dynamic baseline and the alert priorities.
3. Open **Alert Intelligence** to inspect the ranked alerts and audit stream.
4. Use **Experiment Tables** to compute Chapter 4-style tables from your own uploaded dataset, then **Research Evaluation** for detailed metrics.
5. Return to **Detection parameters** and change one value, then create another run for a defensible comparison.''')

st.divider()
st.caption('PLM · Computer Science Department   |   Disk Usage Anomaly Detection Research Platform   |   Academic prototype · Dataset-driven analysis')
