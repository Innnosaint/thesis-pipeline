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


def research_comparison():
    """Direct transcription of Chapter 4, Table 4.4 of the supplied thesis PDF."""
    return pd.DataFrame([
        {'Method':'Static 80%', 'Precision':0.6094,'Recall':0.8667,'F1':0.7156,'TP':39,'FP':25,'FN':6},
        {'Method':'Static 85%', 'Precision':1.0000,'Recall':0.8667,'F1':0.9286,'TP':39,'FP':0,'FN':6},
        {'Method':'Enhanced (suppression on)', 'Precision':1.0000,'Recall':0.6000,'F1':0.7500,'TP':27,'FP':0,'FN':18},
        {'Method':'Hybrid (priority only)', 'Precision':1.0000,'Recall':0.8667,'F1':0.9286,'TP':39,'FP':0,'FN':6},
    ])


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
    ('RESEARCH', ['Methodology', 'Research Evaluation']),
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
        st.markdown('### Verified thesis results (Chapter 4)')
        tiles([
            ('Observations','105,122','Controlled research benchmark','#50b3ee'),
            ('Enhanced precision','100.0%','Suppression on','#46ca9a'),
            ('Enhanced recall','60.0%','Suppression on','#e6b64e'),
            ('Enhanced F1','75.0%','Suppression on','#ed6179'),
        ])
        st.markdown('<div class="info-strip">The four research benchmark figures above come from the uploaded thesis. They are historical controlled-experiment findings, not current server usage or results from a new dataset. Use Research Evaluation for the full comparison.</div>',unsafe_allow_html=True)

elif page=='Dataset Laboratory':
    section('02 / Experimental workspace','Dataset laboratory',
            'Run the original pipeline on your own logs or the provided research samples. Every experiment is stored separately.')
    with st.container(border=True):
        st.markdown('### 1. Select a data source')
        choice=st.radio('Source',['Example CSV dataset','Included full Linux log','Upload custom dataset'],horizontal=True)
        datasets=[]
        if choice=='Example CSV dataset':
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
        else:
            ups=st.file_uploader('Upload log, text, or disk usage CSV files',type=['log','txt','csv'],accept_multiple_files=True,
                help='For CSV use timestamp,disk_usage_value and optional message columns.')
            if ups:
                for f in ups:
                    try:
                        source=f.getvalue()
                        text_lines=parse_upload(f.name,source)
                        datasets.append((f.name,source))
                        st.success(f'{f.name} · {len(text_lines.splitlines()):,} valid input lines')
                    except ValueError as exc: st.error(f'{f.name}: {exc}')
        st.divider()
        st.markdown('### 2. Optional ground-truth evaluation')
        demo_labels=False
        if choice=='Example CSV dataset' and (ROOT/'data/examples/demo_labels.csv').exists():
            demo_labels=st.checkbox('Use accompanying example labels (demonstration only)',value=True)
        labels_file=st.file_uploader('Upload independently labeled observations',type=['csv','jsonl'],
            help='Optional. Must contain timestamp and is_anomaly. This is required for precision, recall and F1.')
        labels_data=labels_file.getvalue() if labels_file else None
        labels_name=labels_file.name if labels_file else ''
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
                    run_folder,result=run_dataset(datasets,labels_data,labels_name,window,min_obs,theta,ceiling,debounce,delta,suppress_benign)
                    st.session_state['run_folder']=str(run_folder)
                st.success(f'Experiment completed · {result["readings"]:,} disk observations · {result["alerts"]:,} delivered alerts. Select Overview, Alert Intelligence, or Research Evaluation.')
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
        st.markdown('### Highest-ranked alerts')
        top=actionable.sort_values('enhanced_alert_score',ascending=False).head(4)
        if len(top):
            cols=st.columns(2,gap='medium')
            for i,(_,rec) in enumerate(top.iterrows()):
                with cols[i%2]:
                    priority=str(rec.get('priority','MEDIUM'))
                    color='#D6869E' if priority=='HIGH' else '#D8B475' if priority=='MEDIUM' else '#52D6BC'
                    dt=rec['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(rec['timestamp']) else 'Timestamp unavailable'
                    st.markdown(f'<div class="alert-card"><div><div class="alert-priority" style="color:{color}">{esc(priority)} PRIORITY</div>'
                                f'<div class="alert-name">{float(rec["disk_usage_value"]):.1f}% disk utilization</div>'
                                f'<div class="alert-meta">{esc(dt)}<br>Context: {esc(rec.get("matched_keyword") or "No signature identified")}</div></div>'
                                f'<div class="alert-score">{float(rec["enhanced_alert_score"]):.2f}</div></div>',unsafe_allow_html=True)
        else: st.info('No delivered alerts under the selected experiment parameters.')
        st.markdown('### Detailed event explorer')
        cat=st.selectbox('Classification',['Delivered alerts','All anomalous events','All readings'])
        view=actionable if cat=='Delivered alerts' else abnormal if cat=='All anomalous events' else df
        filter_priorities=st.multiselect('Priority filter',['HIGH','MEDIUM','LOW'],default=['HIGH','MEDIUM','LOW'])
        if cat!='All readings':view=view.loc[view.priority.isin(filter_priorities)]
        view=view.sort_values('enhanced_alert_score',ascending=False)
        columns=['timestamp','log_source','disk_usage_value','baseline_mu','zscore','classification','triggered_by','priority','matched_keyword','enhanced_alert_score','suppressed']
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
    section('04 / Research evidence','Algorithm performance evaluation',
            'Separate the documented thesis benchmark from new experiment metrics for an accurate defense presentation.')
    st.markdown('<span class="research-label">Chapter 4 / Table 4.4 / Thesis benchmark</span>',unsafe_allow_html=True)
    research=research_comparison()
    tiles([
        ('Enhanced precision','100.0%','With benign suppression','#52D6BC'),
        ('Enhanced recall','60.0%','27 true positives, 18 missed','#75B9EE'),
        ('Enhanced F1','75.0%','Suppression-enabled setting','#D8B475'),
        ('Priority-only F1','92.9%','Post-hoc hybrid ablation','#D6869E'),
    ])
    st.markdown('### Published comparison from the supplied thesis')
    comparison_chart(research)
    pretty=research.copy()
    for col in ['Precision','Recall','F1']:pretty[col]=pretty[col].map(lambda v:f'{v:.2%}')
    st.dataframe(pretty,hide_index=True,use_container_width=True)
    st.markdown('<div class="info-strip"><b>Interpretation:</b> The suppression-enabled approach reduced false positives compared with the 80% rule but missed some labeled anomalies after contextual suppression. The priority-only hybrid matched a static 85% rule on this benchmark. The 85% value was also used to define labels, so these results do not demonstrate generalization to unseen production data. These findings are transcribed from the uploaded thesis, not recalculated by the dashboard.</div>',unsafe_allow_html=True)
    st.divider()
    st.markdown('### Current experiment performance')
    if summary and summary.get('metrics') is not None:
        m=summary['metrics']
        st.markdown(f'<span class="run-pill">RUN {esc(summary["run_id"])}</span>',unsafe_allow_html=True)
        if m['matched'] == 0:st.error('No ground-truth timestamps matched baseline-ready readings. Do not interpret the following zero scores as valid accuracy measurements.')
        else:
            tiles([
                ('Precision',f'{m["precision"]:.1%}',f'{m["tp"]} true-positive alerts','#52D6BC'),
                ('Recall',f'{m["recall"]:.1%}',f'{m["fn"]} false negatives','#75B9EE'),
                ('F1 score',f'{m["f1"]:.1%}','Balance of precision and recall','#D8B475'),
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
                st.markdown('#### Baseline method comparison')
                variants=[('Enhanced (current)',m),('Static 80%',summary.get('static_80_metrics')),('Static 85%',summary.get('static_85_metrics'))]
                vals=[{'Method':name,'Precision':mm['precision'],'Recall':mm['recall'],'F1':mm['f1']} for name,mm in variants if mm]
                if vals:comparison_chart(pd.DataFrame(vals))
            st.caption(f'Matched {m["matched"]:,} baseline-ready timestamps of {m["labels_count"]:,} labeled timestamps; repeated timestamps are evaluated once. Method comparisons here use the same matched labels and warm-up eligibility.')
        st.download_button('Download this run’s evaluation',json.dumps(m,indent=2),file_name='experiment_evaluation.json',mime='application/json')
    elif summary:
        empty('Ground truth not provided','You can inspect anomaly output, but accuracy measures cannot be calculated for this experiment without independently prepared labels.')
    else:
        empty('No current experiment','The comparison above is taken directly from the thesis document. To produce new scores, load a dataset and an independent ground-truth label file.')

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
4. Use **Research Evaluation** to compare the selected experiment with the thesis benchmark.
5. Return to **Detection parameters** and change one value, then create another run for a defensible comparison.''')

st.divider()
st.caption('PLM · Computer Science Department   |   Disk Usage Anomaly Detection Research Platform   |   Academic prototype · Dataset-driven analysis')
