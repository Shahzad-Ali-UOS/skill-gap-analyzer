# data_generator.py
import pandas as pd
import numpy as np

def generate_datasets():
    # 1. Industry Demands Corpus
    industry_jobs = [
        # Cluster 0: Machine Learning / AI
        {"job_id": "JOB-ML-01", "role": "Machine Learning Engineer", "cluster_label": "Machine Learning / AI",
         "skills": "python, scikit-learn, pytorch, pandas, numpy, machine learning, feature engineering, model evaluation, deep learning, git",
         "description": "Develop supervised machine learning pipelines, optimize PyTorch neural network architectures, train predictive models, and validate feature importance using cross validation."},
        {"job_id": "JOB-ML-02", "role": "NLP & LLM Engineer", "cluster_label": "Machine Learning / AI",
         "skills": "python, transformers, pytorch, huggingface, nlp, tf-idf, langchain, text classification, spacy, git",
         "description": "Build production NLP pipelines, fine-tune transformer models, design semantic embeddings, perform text tokenization, and deploy generative AI applications."},
        {"job_id": "JOB-ML-03", "role": "Computer Vision Engineer", "cluster_label": "Machine Learning / AI",
         "skills": "python, opencv, pytorch, cnn, object detection, image processing, numpy, yolo, transfer learning, git",
         "description": "Implement real-time computer vision models using OpenCV, train convolutional neural networks, perform image segmentation, and benchmark edge inference models."},

        # Cluster 1: Backend Engineering
        {"job_id": "JOB-BE-01", "role": "Backend Engineer (FastAPI/Python)", "cluster_label": "Backend Systems",
         "skills": "python, fastapi, rest api, postgresql, redis, async programming, docker, pydantic, sqlalchemy, git",
         "description": "Design asynchronous high-throughput REST APIs using FastAPI, optimize PostgreSQL queries, integrate Redis caching layers, and manage microservice routing."},
        {"job_id": "JOB-BE-02", "role": "Django Web Architect", "cluster_label": "Backend Systems",
         "skills": "python, django, django rest framework, postgresql, celery, redis, relational database, orm, unit testing, git",
         "description": "Architect enterprise web platforms using Django ORM, resolve N+1 database querying bottlenecks, orchestrate Celery background worker queues, and write unit tests."},
        {"job_id": "JOB-BE-03", "role": "Distributed Systems Developer", "cluster_label": "Backend Systems",
         "skills": "python, go, microservices, rabbitmq, grpc, postgresql, distributed systems, docker, rest api, git",
         "description": "Develop resilient microservices with event-driven message brokers, implement gRPC interfaces, handle database transactions, and maintain scalable architectures."},

        # Cluster 2: DevOps & Cloud
        {"job_id": "JOB-DO-01", "role": "Cloud DevOps Engineer", "cluster_label": "Cloud & DevOps",
         "skills": "docker, kubernetes, aws, terraform, ci/cd, linux, bash, github actions, prometheus, monitoring",
         "description": "Automate cloud infrastructure provisioning using Terraform on AWS, configure Kubernetes pod auto-scaling, maintain GitHub Actions CI/CD pipelines, and monitor metrics."},
        {"job_id": "JOB-DO-02", "role": "Site Reliability Engineer", "cluster_label": "Cloud & DevOps",
         "skills": "linux, kubernetes, docker, grafana, prometheus, terraform, bash, python, cloud security, networking",
         "description": "Ensure high availability and fault tolerance of production clusters, monitor system latency with Grafana and Prometheus, and configure container security policies."},

        # Cluster 3: Data Analytics & BI
        {"job_id": "JOB-DA-01", "role": "Data Analyst", "cluster_label": "Data Analytics & BI",
         "skills": "sql, python, power bi, tableau, pandas, data visualization, exploratory data analysis, statistics, excel, reporting",
         "description": "Extract data using complex SQL window functions, perform exploratory data analysis using Pandas, and build interactive executive dashboards in Power BI and Tableau."},
        {"job_id": "JOB-DA-02", "role": "Business Intelligence Analyst", "cluster_label": "Data Analytics & BI",
         "skills": "power bi, dax, sql, data modeling, tableau, reporting, business metrics, etl, excel, data warehousing",
         "description": "Construct enterprise dimensional data models, author optimized DAX calculated measures in Power BI, design automated ETL pipelines, and track KPI revenue trends."}
    ]

    # 2. Intern Profiles Database
    intern_profiles = [
        {"intern_id": "INT-101", "name": "Hamza Tariq", "enrolled_track": "Machine Learning",
         "skills": "python, pandas, numpy, scikit-learn, git",
         "proficiency_levels": "python:4, pandas:4, numpy:3, scikit-learn:3, git:3",
         "completed_hours": 85},
        {"intern_id": "INT-102", "name": "Ayesha Bilal", "enrolled_track": "Backend Systems",
         "skills": "python, django, rest api, postgresql, git",
         "proficiency_levels": "python:4, django:4, rest api:3, postgresql:3, git:3",
         "completed_hours": 92},
        {"intern_id": "INT-103", "name": "Bilal Khan", "enrolled_track": "Data Analytics & BI",
         "skills": "sql, power bi, excel, data visualization",
         "proficiency_levels": "sql:3, power bi:4, excel:4, data visualization:3",
         "completed_hours": 70},
        {"intern_id": "INT-104", "name": "Zainab Arshad", "enrolled_track": "Cloud & DevOps",
         "skills": "linux, bash, git, docker",
         "proficiency_levels": "linux:3, bash:3, git:3, docker:2",
         "completed_hours": 65}
    ]

    df_jobs = pd.DataFrame(industry_jobs)
    df_jobs.to_csv("data/industry_jobs.csv", index=False)

    df_interns = pd.DataFrame(intern_profiles)
    df_interns.to_csv("data/intern_profiles.csv", index=False)

    print("[SUCCESS] Created data/industry_jobs.csv and data/intern_profiles.csv")

if __name__ == "__main__":
    generate_datasets()