import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Prescriptive Course Catalog mapped to common skill gaps
CURRICULUM_CATALOG = {
    "pytorch": {"course": "Deep Learning Specialization with PyTorch", "hours": 24, "level": "Advanced"},
    "deep learning": {"course": "Neural Network Architectures & Backpropagation", "hours": 20, "level": "Advanced"},
    "feature engineering": {"course": "Production Feature Engineering & Scaling", "hours": 12, "level": "Intermediate"},
    "model evaluation": {"course": "Rigorous Cross-Validation & Metric Auditing", "hours": 10, "level": "Foundational"},
    "fastapi": {"course": "Asynchronous High-Throughput APIs with FastAPI", "hours": 18, "level": "Intermediate"},
    "redis": {"course": "In-Memory Caching & Session Stores with Redis", "hours": 14, "level": "Intermediate"},
    "celery": {"course": "Distributed Task Queues & Asynchronous Workers", "hours": 16, "level": "Advanced"},
    "docker": {"course": "Containerization & Multi-Stage Docker Builds", "hours": 15, "level": "Foundational"},
    "kubernetes": {"course": "Kubernetes Pod Orchestration & Helm Charts", "hours": 28, "level": "Advanced"},
    "aws": {"course": "Cloud Architecture & EC2/S3 Deployment Patterns", "hours": 22, "level": "Intermediate"},
    "tableau": {"course": "Enterprise Dashboard Visualizations with Tableau", "hours": 14, "level": "Intermediate"},
    "dax": {"course": "Advanced DAX Calculations & Dimensional Modeling", "hours": 18, "level": "Advanced"},
    "statistics": {"course": "Statistical Hypothesis Testing & A/B Experiments", "hours": 12, "level": "Foundational"}
}

class SkillClusteringEngine:
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[\w-]+\b', stop_words='english')
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.pca = PCA(n_components=2, random_state=42)
        self.industry_df = None
        self.industry_tfidf = None
        self.pca_coords = None
        self.cluster_names = {}

    def fit(self, industry_df: pd.DataFrame):
        self.industry_df = industry_df.copy()
        # Combine skills and description for rich semantic representation
        corpus = (industry_df['skills'] + " " + industry_df['description']).tolist()
        self.industry_tfidf = self.vectorizer.fit_transform(corpus)
        
        # Fit K-Means
        self.kmeans.fit(self.industry_tfidf)
        self.industry_df['cluster'] = self.kmeans.labels_

        # Map cluster IDs to human-readable names based on modal cluster_label
        for c in range(self.n_clusters):
            matched_labels = self.industry_df[self.industry_df['cluster'] == c]['cluster_label']
            if not matched_labels.empty:
                self.cluster_names[c] = matched_labels.mode()[0]
            else:
                self.cluster_names[c] = f"Cluster {c}"

        # Fit 2D PCA projection for interactive scatter plotting
        self.pca_coords = self.pca.fit_transform(self.industry_tfidf.toarray())
        self.industry_df['pca_x'] = self.pca_coords[:, 0]
        self.industry_df['pca_y'] = self.pca_coords[:, 1]

    def analyze_intern(self, intern_skills_str: str, target_cluster_id: int = None) -> Dict:
        """Projects an intern into the semantic cluster space and identifies skill deficiencies."""
        intern_vec = self.vectorizer.transform([intern_skills_str])
        intern_pca = self.pca.transform(intern_vec.toarray())[0]

        # Determine best-matching cluster if none specified
        sims_to_centers = cosine_similarity(intern_vec, self.kmeans.cluster_centers_)[0]
        assigned_cluster = int(np.argmax(sims_to_centers)) if target_cluster_id is None else target_cluster_id
        cluster_affinity = round(float(sims_to_centers[assigned_cluster]) * 100, 1)

        # Retrieve benchmark skills for this cluster
        cluster_jobs = self.industry_df[self.industry_df['cluster'] == assigned_cluster]
        benchmark_skills = set()
        for s_list in cluster_jobs['skills'].str.split(', '):
            benchmark_skills.update([s.strip().lower() for s in s_list])

        # Intern skill set
        intern_skills = set([s.strip().lower() for s in intern_skills_str.split(',') if s.strip()])

        matched = sorted(list(benchmark_skills.intersection(intern_skills)))
        missing = sorted(list(benchmark_skills.difference(intern_skills)))

        readiness_score = round((len(matched) / max(1, len(benchmark_skills))) * 100, 1)

        # Formulate prescriptive course recommendations
        prescribed_training = []
        for gap_skill in missing:
            if gap_skill in CURRICULUM_CATALOG:
                info = CURRICULUM_CATALOG[gap_skill]
                prescribed_training.append({
                    "skill": gap_skill.title(),
                    "course": info["course"],
                    "hours": info["hours"],
                    "level": info["level"]
                })
            else:
                prescribed_training.append({
                    "skill": gap_skill.title(),
                    "course": f"Foundational & Applied {gap_skill.title()} Lab",
                    "hours": 12,
                    "level": "Intermediate"
                })

        return {
            "assigned_cluster": assigned_cluster,
            "cluster_name": self.cluster_names.get(assigned_cluster, f"Cluster {assigned_cluster}"),
            "cluster_affinity": cluster_affinity,
            "readiness_score": readiness_score,
            "pca_x": float(intern_pca[0]),
            "pca_y": float(intern_pca[1]),
            "matched_skills": matched,
            "missing_skills": missing,
            "total_benchmark_skills": len(benchmark_skills),
            "prescribed_training": prescribed_training
        }