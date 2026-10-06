# 🧭 Industrial Skill Gap & Competency Analysis Engine

An end-to-end NLP and unsupervised clustering system built to benchmark intern competencies against real-time industry demands. The platform projects candidate profiles and industry role requirements into a shared latent semantic space, detects technical deficiencies across key tech archetypes, and prescribes prioritized upskilling roadmaps.

---

## 📌 Project Overview

- **Unsupervised Role Clustering:** Applies **TF-IDF** vectorization and **K-Means clustering** ($k=4$) across hundreds of industry job descriptions to categorize tech disciplines (Machine Learning/AI, Backend Systems, Cloud/DevOps, and Data Analytics/BI).
- **2D Latent Space Projection:** Leverages **Principal Component Analysis (PCA)** to project high-dimensional skill spaces into interactive 2D coordinates, mapping the candidate's exact location relative to industry clusters.
- **Skill Deficiency Delta:** Computes set-theoretic and cosine distance deltas to separate verified candidate strengths from high-priority skill gaps.
- **Prescriptive Upskilling Roadmap:** Automatically pairs identified skill deficiencies with targeted training modules, estimated completion hours, and course difficulty ratings.
- **Audit Export:** 1-click generation of candidate evaluation scorecards in Markdown for internship mentors.

---

## 🛠️️ Tech Stack & Architecture

- **Language:** Python 3.10+
- **Machine Learning & NLP:** Scikit-Learn (`TfidfVectorizer`, `KMeans`, `PCA`, `cosine_similarity`)
- **Data Engineering:** Pandas, NumPy
- **Dashboard & Visualizations:** Streamlit, Plotly (`express`, `graph_objects`)
- **Model Serialization:** Joblib

---

## 📂 Repository Structure

```text
skill_gap_analyzer/
├── data/
│   ├── industry_jobs.csv       # Corpus of industry roles, skills, and descriptions
│   └── intern_profiles.csv     # Candidate competency profiles
├── models/
│   └── clustering_artifacts.pkl # Fitted vectorizer, KMeans, and PCA pipeline
├── engine/
│   ├── __init__.py
│   └── cluster_engine.py       # Core NLP vectorization, clustering, and delta logic
├── app.py                      # Interactive Streamlit cockpit
├── data_generator.py           # Synthetic dataset generator
├── train.py                    # Model training and artifact serialization script
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation