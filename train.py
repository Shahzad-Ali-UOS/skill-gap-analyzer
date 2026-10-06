# train.py
import joblib
import pandas as pd
from engine.cluster_engine import SkillClusteringEngine

def train():
    print("=" * 60)
    print("      TRAINING NLP & K-MEANS SKILL CLUSTERING ENGINE")
    print("=" * 60)

    # 1. Load Industry Jobs
    df_jobs = pd.read_csv("data/industry_jobs.csv")
    print(f"Loaded {len(df_jobs)} industry job profiles.")

    # 2. Fit Engine
    engine = SkillClusteringEngine(n_clusters=4)
    engine.fit(df_jobs)
    print("Fitted TF-IDF Vectorizer, K-Means (k=4), and 2D PCA Projections.")

    for cid, cname in engine.cluster_names.items():
        job_count = len(engine.industry_df[engine.industry_df['cluster'] == cid])
        print(f"  • Cluster {cid}: {cname} ({job_count} roles)")

    # 3. Save Serialized Engine Artifact
    joblib.dump(engine, "models/clustering_artifacts.pkl")
    print("\n[SUCCESS] Model artifacts saved to models/clustering_artifacts.pkl")

    # 4. Quick Self-Test
    sample_intern_skills = "python, pandas, numpy, scikit-learn, git"
    res = engine.analyze_intern(sample_intern_skills)
    print("\n--- Self-Test Verification ---")
    print(f"Input Skills     : {sample_intern_skills}")
    print(f"Assigned Cluster : {res['cluster_name']}")
    print(f"Affinity Score   : {res['cluster_affinity']}%")
    print(f"Readiness Score  : {res['readiness_score']}%")
    print(f"Missing Gaps ({len(res['missing_skills'])}): {', '.join(res['missing_skills'][:4])}...")
    print(f"Training Courses : {len(res['prescribed_training'])} modules recommended.")

if __name__ == "__main__":
    train()