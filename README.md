

🌍 Problem & Vision

AZCON operates across multiple national-level service providers:

📶 Telecom — Aztelekom
✈️ Aviation — AZAL
📦 Postal — Azərpoçt
🚆 Railway — ADY
🆔 Digital Identity — Azİntelekom

Today, these organizations function as independent data silos.

Each company:

Sees the customer only within its own service scope
Uses different identification systems
Has limited or no data interoperability
Optimizes decisions locally rather than ecosystem-wide

As a result:

The same individual exists as multiple disconnected “customers” across systems

⚠️ Core Problems

This fragmentation leads to:

Incomplete understanding of customer behavior
Missed cross-company revenue opportunities
Inefficient and duplicated data processing
Isolated churn prediction and risk analysis
Disconnected and suboptimal customer experiences
🎯 Vision

The goal of AZCON is to establish:

A unified data foundation (central data lake) and a shared customer perspective across all participating entities

Instead of siloed analytics, AZCON enables:

Cross-domain data integration
A consistent and unified customer identity
Ecosystem-level intelligence and decision-making
🧠 Target State

AZCON aims to build:

Single Source of Truth → A unified 360° behavioral customer profile across industries

This profile combines:

Telecom usage patterns
Travel behavior (air & rail)
Logistics and parcel activity
Mobility signals
Digital identity interactions
⚡ Outcome

With this approach:

Customer behavior is modeled holistically, not in isolation
Companies move from local optimization to ecosystem intelligence
Data becomes a shared strategic asset rather than a fragmented resource
New cross-company use cases and revenue streams become possible
💡 Key Principle

The objective is not just to aggregate data
but to transform fragmented systems into a unified data ecosystem[04:15, 4/27/2026] Kamal Holberton: ## 🌍 Vision & Purpose

*AzCon (Persona Engine)* is a large-scale data intelligence platform designed to eliminate data silos across multiple national-level service providers:

- 📶 Telecom — Aztelekom  
- ✈️ Aviation — AZAL  
- 📦 Postal — Azərpoçt  
- 🚆 Railway — ADY  
- 🆔 Digital Identity — Azİntelekom  

The core vision is to build a:

> *Single Source of Truth → A unified 360° behavioral customer profile across industries*

Instead of analyzing customers in isolation, AzCon models *cross-domain behavioral patterns*, enabling:

- Accurate churn forecasting  
- Cross-company monetization  
- Real-time decision intelligence  

---

## 🧠 Data & Machine Learning Architecture

AzCon is built as a *modular data intelligence pipeline*, transforming fragmented raw data into business-critical insights.

---

## 🧩 1. Data Layer — Multi-Source Integration

### 🔗 Data Sources

The system ingests structured datasets from 5 independent domains:

- Telecom usage & billing patterns  
- Airline travel frequency & loyalty  
- Parcel & logistics behavior  
- Railway mobility data  
- Digital identity & authentication signals  

Each dataset is:

- Cleaned (handling missing values)  
- Standardized (schema alignment)  
- Linked via a *global unique key (phone_number)*  

---

## 🏗️ 2. Unified Feature Store (Core Layer)

At the heart of AzCon lies a *centralized Feature Store*, acting as:

> 🧠 *The single computational layer for all ML models and analytics*

### ⚙️ Key Characteristics

- Columnar structured dataset (feature_store.csv)  
- Fully denormalized (no joins required at inference time)  
- Zero missing values (after preprocessing)  
- Optimized for ML pipelines  

---

## 🧪 Feature Engineering Strategy

Instead of using noisy raw data, AzCon extracts *high-signal composite features*:

### 🔑 Core Engineered Features

- *revenue_score* → Total ecosystem value  
- *risk_score* → Early churn signal  
- *digital_score* → Digital engagement level  
- *loyalty_score* → Retention & loyalty strength  
- *mobility_score* → Travel & movement behavior  
- *cross_sell_potential* → Expansion opportunity score  

---

## 🧮 Dimensionality Reduction Logic

Instead of 90+ raw features:

> *We reduce them to 10–12 high-impact, interpretable features*

Benefits:

- Better generalization  
- Reduced noise  
- Improved interpretability  

---

## 🤖 3. Machine Learning Layer

Core pipeline:


### 🔍 3.1 Churn Prediction (Random Forest)

*Why Random Forest?*

- Handles non-linear patterns  
- Robust to noise  
- Strong performance on tabular data  

*Model Setup:*

- ~300 trees  
- Controlled depth to prevent overfitting  
- Stratified train-test split  

*Output:*

- churn_score (0–1)  
- Risk tiers:
  - 🟢 Loyal  
  - 🟡 Medium Risk  
  - 🔴 High Risk  

👉 Used for retention prioritization  

---

### 🔗 3.2 Customer Segmentation (K-Means)

*Why K-Means?*

- Fast and scalable  
- Easy to interpret  

*Pipeline:*

- Feature scaling (StandardScaler)  
- Clustering into 5 segments  

*Segment Examples:*

- Digital Power Users  
- High Value Loyalists  
- High Risk Users  
- Business Personas  
- Standard Users  

👉 Used for personalization  

---

### 📈 3.3 CLTV Approximation

Based on:

- Spending  
- Tenure  
- Multi-service usage  

→ Generates forward-looking customer value  

---

### 🎯 3.4 Cross-Sell Potential

Calculated using:

- Multi-company activity  
- Behavioral overlap  
- Engagement intensity  

→ Produces cross-sell score  

---

## ⚡ 4. Recommendation Engine (Decision Layer)

> 🔥 Behavior-driven cross-company intelligence

---

### 🧠 Logic

System evaluates:

- Behavior patterns  
- Risk score  
- Segment  
- Cross-company signals  

Then generates:

> 🎯 *Next Best Action (NBA)*

---

### 🔄 Example Scenarios

*✈️ Travel Detected (AZAL)*  
→ 📶 Internet freeze (Aztelekom)  
→ 🚆 Airport transfer (ADY)  

*📦 High Parcel Activity (Azərpoçt)*  
→ ✈️ Business travel offers (AZAL)  

*🧠 High Digital Activity (Azİntelekom)*  
→ 📶 Smart home / cloud services  

---

### 💡 Key Idea

> Not product-based recommendations  
> but *behavior-based solutions*

---

## 🧠 5. Generative AI Layer

### ⚙️ Architecture

- ML outputs → structured features  
- Injected into prompt templates  
- Processed via LLM (Groq API)  

### Output

- Context-aware campaigns  
- Natural language offers  
- JSON structured recommendations  

---

## 💻 6. System Interface (SaaS CRM)

### Tech Stack

- *Frontend:* Streamlit  
- *Visualization:* Plotly  
- *Backend:* Python (Pandas, NumPy, Scikit-learn)  
- *AI:* Groq API  

---

### 🖥️ Features

- 📊 Global analytics dashboard  
- 👤 360° customer profile  
- ⚡ AI campaign generator  
- 📁 CSV export  

---

## 📊 Pipeline Scale (Demo)

- 👥 2,000 users  
- 🧠 40+ ML features  
- 🔗 5 integrated systems  
- 🎯 28,000+ recommendations  
- 📈 ~14 recommendations per user  

---

## 💼 Business Impact

Transforms companies into:

> 🚀 Data-driven ecosystems

### Value:

- Cross-company revenue  
- Better targeting  
- Reduced churn  
- Increased customer lifetime value  

---

## 🔐 Strategic Advantage

> AzCon is not just a model — it is an infrastructure layer

---

## 🏁 Final Statement

> *From data → to behavior → to revenue*[04:15, 4/27/2026] Kamal Holberton: ## 🌍 Vision & Purpose

*AzCon (Persona Engine)* is a large-scale data intelligence platform designed to eliminate data silos across multiple national-level service providers:

- 📶 Telecom — Aztelekom  
- ✈️ Aviation — AZAL  
- 📦 Postal — Azərpoçt  
- 🚆 Railway — ADY  
- 🆔 Digital Identity — Azİntelekom  

The core vision is to build a:

> *Single Source of Truth → A unified 360° behavioral customer profile across industries*

Instead of analyzing customers in isolation, AzCon models *cross-domain behavioral patterns*, enabling:

- Accurate churn forecasting  
- Cross-company monetization  
- Real-time decision intelligence  

---

## 🧠 Data & Machine Learning Architecture

AzCon is built as a *modular data intelligence pipeline*, transforming fragmented raw data into business-critical insights.

---

## 🧩 1. Data Layer — Multi-Source Integration

### 🔗 Data Sources

The system ingests structured datasets from 5 independent domains:

- Telecom usage & billing patterns  
- Airline travel frequency & loyalty  
- Parcel & logistics behavior  
- Railway mobility data  
- Digital identity & authentication signals  

Each dataset is:

- Cleaned (handling missing values)  
- Standardized (schema alignment)  
- Linked via a *global unique key (phone_number)*  

---

## 🏗️ 2. Unified Feature Store (Core Layer)

At the heart of AzCon lies a *centralized Feature Store*, acting as:

> 🧠 *The single computational layer for all ML models and analytics*

### ⚙️ Key Characteristics

- Columnar structured dataset (feature_store.csv)  
- Fully denormalized (no joins required at inference time)  
- Zero missing values (after preprocessing)  
- Optimized for ML pipelines  

---

## 🧪 Feature Engineering Strategy

Instead of using noisy raw data, AzCon extracts *high-signal composite features*:

### 🔑 Core Engineered Features

- *revenue_score* → Total ecosystem value  
- *risk_score* → Early churn signal  
- *digital_score* → Digital engagement level  
- *loyalty_score* → Retention & loyalty strength  
- *mobility_score* → Travel & movement behavior  
- *cross_sell_potential* → Expansion opportunity score  

---

## 🧮 Dimensionality Reduction Logic

Instead of 90+ raw features:

> *We reduce them to 10–12 high-impact, interpretable features*

Benefits:

- Better generalization  
- Reduced noise  
- Improved interpretability  

---

## 🤖 3. Machine Learning Layer

Core pipeline:
[04:15, 4/27/2026] Kamal Holberton: ---

### 🔍 3.1 Churn Prediction (Random Forest)

*Why Random Forest?*

- Handles non-linear patterns  
- Robust to noise  
- Strong performance on tabular data  

*Model Setup:*

- ~300 trees  
- Controlled depth to prevent overfitting  
- Stratified train-test split  

*Output:*

- churn_score (0–1)  
- Risk tiers:
  - 🟢 Loyal  
  - 🟡 Medium Risk  
  - 🔴 High Risk  

👉 Used for retention prioritization  

---

### 🔗 3.2 Customer Segmentation (K-Means)

*Why K-Means?*

- Fast and scalable  
- Easy to interpret  

*Pipeline:*

- Feature scaling (StandardScaler)  
- Clustering into 5 segments  

*Segment Examples:*

- Digital Power Users  
- High Value Loyalists  
- High Risk Users  
- Business Personas  
- Standard Users  

👉 Used for personalization  

---

### 📈 3.3 CLTV Approximation

Based on:

- Spending  
- Tenure  
- Multi-service usage  

→ Generates forward-looking customer value  

---

### 🎯 3.4 Cross-Sell Potential

Calculated using:

- Multi-company activity  
- Behavioral overlap  
- Engagement intensity  

→ Produces cross-sell score  

---

## ⚡ 4. Recommendation Engine (Decision Layer)

> 🔥 Behavior-driven cross-company intelligence

---

### 🧠 Logic

System evaluates:

- Behavior patterns  
- Risk score  
- Segment  
- Cross-company signals  

Then generates:

> 🎯 *Next Best Action (NBA)*

---

### 🔄 Example Scenarios

*✈️ Travel Detected (AZAL)*  
→ 📶 Internet freeze (Aztelekom)  
→ 🚆 Airport transfer (ADY)  

*📦 High Parcel Activity (Azərpoçt)*  
→ ✈️ Business travel offers (AZAL)  

*🧠 High Digital Activity (Azİntelekom)*  
→ 📶 Smart home / cloud services  

---

### 💡 Key Idea

> Not product-based recommendations  
> but *behavior-based solutions*

---

## 🧠 5. Generative AI Layer

### ⚙️ Architecture

- ML outputs → structured features  
- Injected into prompt templates  
- Processed via LLM (Groq API)  

### Output

- Context-aware campaigns  
- Natural language offers  
- JSON structured recommendations  

---

## 💻 6. System Interface (SaaS CRM)

### Tech Stack

- *Frontend:* Streamlit  
- *Visualization:* Plotly  
- *Backend:* Python (Pandas, NumPy, Scikit-learn)  
- *AI:* Groq API  

---

### 🖥️ Features

- 📊 Global analytics dashboard  
- 👤 360° customer profile  
- ⚡ AI campaign generator  
- 📁 CSV export  

---

## 📊 Pipeline Scale (Demo)

- 👥 2,000 users  
- 🧠 40+ ML features  
- 🔗 5 integrated systems  
- 🎯 28,000+ recommendations  
- 📈 ~14 recommendations per user  

---

## 💼 Business Impact

Transforms companies into:

> 🚀 Data-driven ecosystems

### Value:

- Cross-company revenue  
- Better targeting  
- Reduced churn  
- Increased customer lifetime value  

---

## 🔐 Strategic Advantage

> AzCon is not just a model — it is an infrastructure layer

---

## 🏁 Final Statement

> *From data → to behavior → to revenue*
