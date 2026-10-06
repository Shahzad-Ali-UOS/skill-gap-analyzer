import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from engine.cluster_engine import SkillClusteringEngine

st.set_page_config(
    page_title="Skill Gap Analysis & Upskilling Engine | internee.pk",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern UI Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .hero-banner {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.12), rgba(99, 102, 241, 0.12));
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }

    .metric-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.2);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #38BDF8;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    .course-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-left: 4px solid #38BDF8;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model Artifacts & Data
# ---------------------------------------------------------
@st.cache_resource
def load_engine():
    return joblib.load("models/clustering_artifacts.pkl")

@st.cache_data
def load_datasets():
    interns_df = pd.read_csv("data/intern_profiles.csv")
    jobs_df = pd.read_csv("data/industry_jobs.csv")
    return interns_df, jobs_df

try:
    engine: SkillClusteringEngine = load_engine()
    interns_df, jobs_df = load_datasets()
except Exception as e:
    st.error(f"⚠️ Error loading artifacts. Please run `python train.py` first. Details: {e}")
    st.stop()

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/compass--v1.png", width=64)
    st.title("Skill Gap Studio")
    st.caption("NLP & K-Means Industrial Demand Benchmarking")
    st.markdown("---")

    mode = st.radio("Candidate Ingestion:", ["Select Registered Intern", "Custom Skill Input"])

    if mode == "Select Registered Intern":
        intern_options = [f"{row['name']} ({row['enrolled_track']})" for _, row in interns_df.iterrows()]
        sel_idx = st.selectbox("Select Candidate:", range(len(intern_options)), format_func=lambda x: intern_options[x])
        selected_intern = interns_df.iloc[sel_idx]
        candidate_name = selected_intern["name"]
        candidate_skills = selected_intern["skills"]
        enrolled_track = selected_intern["enrolled_track"]
    else:
        candidate_name = st.text_input("Candidate Name:", value="Shahzad Ali")
        enrolled_track = st.selectbox("Intended Career Track:", list(engine.cluster_names.values()))
        candidate_skills = st.text_area(
            "Enter Skills (comma-separated):",
            value="python, pandas, scikit-learn, git, streamlit, xgboost",
            height=100
        )

    st.markdown("---")
    st.subheader("Target Industry Benchmark")
    cluster_options = {cid: cname for cid, cname in engine.cluster_names.items()}
    cluster_override = st.selectbox(
        "Industry Archetype Target:",
        options=list(cluster_options.keys()),
        format_func=lambda x: cluster_options[x]
    )

    analyze_btn = st.button("⚡ Run Skill Audit", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Execution & Gap Extraction
# ---------------------------------------------------------
analysis = engine.analyze_intern(candidate_skills, target_cluster_id=cluster_override)

readiness = analysis["readiness_score"]
affinity = analysis["cluster_affinity"]
matched_skills = analysis["matched_skills"]
missing_skills = analysis["missing_skills"]
training_courses = analysis["prescribed_training"]

# ---------------------------------------------------------
# Hero Banner & KPI Ribbon
# ---------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <h2 style="margin: 0; color: #F8FAFC;">🧭 Industrial Skill Gap & Competency Analysis</h2>
    <p style="margin: 6px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
        Benchmarking <b>{candidate_name}</b> against the <b>{analysis['cluster_name']}</b> industry archetype via NLP & K-Means clustering.
    </p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-sub">Role Readiness</div>
        <div class="metric-val" style="color: {'#10B981' if readiness >= 65 else '#F59E0B'};">{readiness}%</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-sub">Cluster Semantic Affinity</div>
        <div class="metric-val" style="color: #38BDF8;">{affinity}%</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-sub">Verified Competencies</div>
        <div class="metric-val" style="color: #4ADE80;">{len(matched_skills)} <span style="font-size: 1rem; color: #64748B;">/ {analysis['total_benchmark_skills']}</span></div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-sub">Deficiency Gaps</div>
        <div class="metric-val" style="color: #F87171;">{len(missing_skills)}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Visualizations: 2D PCA Cluster Map & Deficiency Delta
# ---------------------------------------------------------
col_scatter, col_delta = st.columns([1.3, 1.2])

with col_scatter:
    st.subheader("🗺️ 2D Semantic Space Projection (PCA)")
    
    # Plot industry benchmarks
    df_plot = engine.industry_df.copy()
    fig_pca = px.scatter(
        df_plot,
        x="pca_x",
        y="pca_y",
        color="cluster_label",
        hover_data=["role", "skills"],
        title="Industry Tech Clusters & Candidate Placement",
        labels={"pca_x": "Latent Skill Dimension 1", "pca_y": "Latent Skill Dimension 2"},
        color_discrete_sequence=["#38BDF8", "#A855F7", "#F59E0B", "#10B981"]
    )
    
    # Add candidate coordinate
    fig_pca.add_trace(go.Scatter(
        x=[analysis["pca_x"]],
        y=[analysis["pca_y"]],
        mode="markers+text",
        marker=dict(symbol="star", size=18, color="#EF4444", line=dict(width=2, color="#FFFFFF")),
        name="Candidate Vector",
        text=[f"★ {candidate_name}"],
        textposition="top center",
        textfont=dict(color="#F8FAFC", size=12)
    ))
    
    fig_pca.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)"
    )
    st.plotly_chart(fig_pca, use_container_width=True)

with col_delta:
    st.subheader("📊 Competency vs. Deficiency Matrix")
    
    # Bar chart comparing verified skills vs gaps
    status_df = pd.DataFrame({
        "Skill": [s.title() for s in matched_skills] + [s.title() for s in missing_skills],
        "Status": ["Verified Match"] * len(matched_skills) + ["Deficiency Gap"] * len(missing_skills),
        "Score": [1] * len(matched_skills) + [-1] * len(missing_skills)
    })
    
    fig_bar = px.bar(
        status_df,
        x="Score",
        y="Skill",
        color="Status",
        orientation="h",
        color_discrete_map={"Verified Match": "#10B981", "Deficiency Gap": "#F87171"},
        title="Skill Verification Status (Cluster Benchmark)"
    )
    fig_bar.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showticklabels=False, title=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# Prescriptive Upskilling Roadmap
# ---------------------------------------------------------
st.subheader("🎯 Prescriptive Upskilling Roadmap")
st.caption("Automated module recommendations mapped specifically to eliminate identified skill deficiencies.")

if training_courses:
    cols = st.columns(2)
    for idx, course in enumerate(training_courses):
        with cols[idx % 2]:
            st.markdown(f"""
            <div class="course-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 700; color: #38BDF8; font-size: 0.85rem;">MODULE {idx+1}: {course['skill'].upper()}</span>
                    <span style="font-size: 0.75rem; color: #94A3B8; border: 1px solid #334155; padding: 2px 8px; border-radius: 9999px;">{course['level']}</span>
                </div>
                <div style="font-size: 1.05rem; font-weight: 600; color: #F8FAFC; margin-bottom: 4px;">
                    {course['course']}
                </div>
                <div style="color: #64748B; font-size: 0.85rem;">
                    ⏱️ Estimated Effort: <b>{course['hours']} hours</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.success("🎉 No deficiencies detected! Candidate covers 100% of the industry benchmark skills.")

# ---------------------------------------------------------
# Export Audit Report
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📥 Export Skill Gap Audit")

audit_md = f"""# Industrial Skill Gap Audit Report - internee.pk
**Candidate:** {candidate_name}  
**Target Archetype:** {analysis['cluster_name']}  
**Readiness Score:** {readiness}%  
**Semantic Affinity:** {affinity}%  

---
## Competency Assessment
- **Verified Benchmark Skills ({len(matched_skills)}):** {', '.join(matched_skills)}
- **Identified Deficiency Gaps ({len(missing_skills)}):** {', '.join(missing_skills)}

---
## Prescriptive Training Roadmap
"""
for idx, c in enumerate(training_courses, 1):
    audit_md += f"{idx}. **{c['course']}** | Focus: {c['skill']} | Effort: {c['hours']}h | Level: {c['level']}\n"

st.download_button(
    label="📄 Download Candidate Audit (.md)",
    data=audit_md,
    file_name=f"Skill_Audit_{candidate_name.replace(' ', '_')}_{analysis['cluster_name'].replace(' ', '_')}.md",
    mime="text/markdown"
)