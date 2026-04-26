

## 🚀 Vision & Purpose
**Persona** is a comprehensive, centralized data intelligence platform designed to tear down data silos across multiple large-scale organizations (e.g., Aztelekom, AzAL, Azerpoct, ADY, AzInTelekom). 

The primary vision of the project is to build a **Single Source of Truth (360° Customer View)** by merging cross-domain consumer behaviors—spanning telecommunications, postal services, domestic transportation, and aviation. By interpreting this merged data through advanced Machine Learning and Generative AI, **Persona** aims to forecast churn risk, calculate Customer Lifetime Value (CLTV), and autonomously recommend the "Next Best Action" to marketing and retention units.

## 🧠 Data & Machine Learning Architecture
The true power of **Persona** lies in its sophisticated, data-driven backend. The analytical engine has been built and trained on **hundreds of thousands of rows of complex consumer transaction logs and behavior traces**.

### 1. Data Processing & Feature Engineering (Data Analytics)
- **Unified Feature Store:** Raw datasets from 5 distinct corporate systems are ingested, cleaned, and merged using **Pandas** and **NumPy**. 
- **Dimensionality Reduction:** Extracting critical KPIs such as *Travel Activity, Digital Engagement, and Loyalty Metrics* leveraging complex aggregations.
- **RFM Analysis:** Advanced Recency-Frequency-Monetary calculations used to derive baseline financial and engagement scores.

### 2. Predictive Machine Learning Models & Pipeline
The core data architecture operates on a streamlined robust pipeline:
**`Feature Store → ML Models → Segments → Predictions → Recommendations`**

Our pre-computed inference pipelines (`feature_store.csv`, `recommendations.csv`) are powered by heavily trained ML models:
- **Churn Prediction (Random Forest):** Evaluates drop-off probabilities across different services. The model is trained on historical retention data, yielding a robust `churn_score` (0.0 to 1.0) and segmenting users into precise `churn_tiers`.
- **Customer Segmentation (K-Means):** Unsupervised learning algorithms group customers into actionable clusters and segments (e.g., *"Power User", "Low Activity", "Ecosystem Champion"*).
- **CLTV Forecasting (Regression Trees):** Projects future revenue generation potential from users up to 12 months ahead based on holistic ecosystem engagement.
- **Cross-Sell Potential (Gradient Boosting):** A multi-label classification/scoring approach evaluating the probability of an existing user organically adopting a parallel service.

### 3. Generative AI Action Engine
- Real-time LLM integration using **Groq API** (powered by open-source extreme-scale models like `gpt-oss-120b`). 
- Dynamic Prompting feeds the customer's exact ML-generated KPIs into the LLM logic, instantly generating highly personalized, native, and contextual marketing campaigns (Next Best Action recommendations) in structured JSON format.

## 💻 Tech Stack & UI/UX Experience
The dashboard operates purely as a modern **SaaS-grade CRM Platform**, abstracting analytical complexity into a clean, intuitive, lightning-fast native interface.
- **Frontend / App Framework:** Streamlit (heavily customized with raw CSS injections for a distinct, native dark-mode enterprise feel).
- **Data Visualization:** Plotly (Plotly Express & Graph_Objects) for dynamic, interactive scatter plots (Intervention Maps) and KPI distribution charts.
- **AI Processing:** Groq inference SDK for ultra-fast, zero-latency GenAI responses.

## 🛠️ Key Dashboard Features
- **Global Overview Analytics:** Intervention mapping to instantly pinpoint high-value users on the verge of churning.
- **Individual CRM Profile (360° View):** Deep-dive UI presenting a specific customer's digital footprint, risk factors, CLTV, and activity progress trackers.
- **AI-Powered Campaign Generator:** One-click dynamic recommendation creator synthesizing ML scores with LLM creativity.
- **Robust Filtering & CSV Exports:** Instant isolation of subsets (e.g., "High Risk, Low Digital Activity") with native one-click export into complete CSV database files for targeted execution.

