# System Architecture - Mermaid Diagrams

## 1. High-Level Component Diagram

```mermaid
graph TB
    subgraph "User Groups"
        direction LR
        P["👤 Patients"]
        C["👨‍⚕️ Clinicians"]
        A["🔧 Admins"]
        PH["📊 Public Health"]
    end

    subgraph "Client Layer"
        direction LR
        WEB["🌐 Web Interface"]
        CLI["⌨️ Terminal/CLI"]
        API["📱 Mobile App"]
    end

    subgraph "FastAPI Backend"
        direction TB
        EPS["📍 REST Endpoints<br/>POST /predict-risk-raw<br/>POST /risk-explanation-raw<br/>GET /risk-explanations<br/>POST /recommend-plan"]
        
        subgraph "Processing Pipeline"
            direction LR
            VAL["✓ Input Validation<br/>& Preprocessing"]
            PRED["🤖 Prediction Engine<br/>RandomForest ×3"]
            EXP["💡 Explanation Engine<br/>Top Factors<br/>Risk Bands<br/>Warnings"]
            REC["🍽️ Recommendation<br/>Engine"]
        end
        
        AUD["📝 Audit & Logging<br/>risk_explanation_log"]
    end

    subgraph "Data Layer"
        direction TB
        DB[("🗄️ SQLite Database<br/>app_data.db")]
        MODELS["🧠 ML Models<br/>(artifacts/)<br/>CKD<br/>Hypertension<br/>Diabetes"]
        TRAIN["📚 Training Data<br/>(1.4M records)<br/>3 preprocessed CSVs"]
    end

    subgraph "External Systems"
        direction LR
        EHR["🏥 EHR Systems<br/>(Future FHIR)"]
        FS["📁 File System<br/>CSV, JSON"]
    end

    P --> WEB
    C --> API
    A --> CLI
    PH --> API

    WEB --> EPS
    CLI --> EPS
    API --> EPS

    EPS --> VAL
    VAL --> PRED
    PRED --> EXP
    EXP --> REC
    EXP --> AUD
    REC --> AUD
    AUD --> DB

    PRED --> MODELS
    MODELS --> TRAIN

    DB -.->|Query| AUD
    EPS -.->|Retrieve| DB

    EPS -.->|Export| EHR
    EPS -.->|Import| FS
```

## 2. Data Flow: Explain-Raw Prediction

```mermaid
sequenceDiagram
    participant User as Patient/Clinician
    participant API as FastAPI Endpoint
    participant VAL as Input Validator
    participant PRED as Prediction Engine
    participant EXP as Explanation Engine
    participant AUD as Audit Service
    participant DB as SQLite Database
    
    User->>API: POST /risk-explanation-raw<br/>(raw patient features)
    
    API->>VAL: Validate & normalize features
    VAL-->>API: Preprocessed feature array
    
    API->>PRED: Predict risk score & class<br/>(load RandomForest model)
    PRED-->>API: Score (0.0-1.0) + class label
    
    API->>EXP: Generate explanation<br/>(top factors, risk band, text)
    EXP-->>API: RiskExplanationRawResponse
    
    API->>AUD: Log prediction with full payload
    AUD->>DB: INSERT into risk_explanation_log<br/>(timestamp, source, JSON)
    DB-->>AUD: Confirmed
    AUD-->>API: Logged
    
    API-->>User: Return risk score + explanation<br/>(top factors, warnings, text)
    
    Note over User,DB: Prediction is now retrievable via<br/>GET /risk-explanations?disease=X&source=api
```

## 3. Database Schema & Relationships

```mermaid
erDiagram
    RISK_PROFILE ||--o{ RISK_EXPLANATION_LOG : "has_history"
    RISK_PROFILE ||--o{ DIET_PLAN : "receives"
    DIET_PLAN ||--o{ MEAL_LOG : "tracks"
    RISK_PROFILE ||--o{ ALERT : "generates"
    RISK_PROFILE ||--o{ RISK_EXPLANATION_LOG : "creates"

    RISK_PROFILE {
        int id PK
        string user_id FK
        string disease
        string model_name
        int predicted_class
        float risk_score
        timestamp created_at
    }

    RISK_EXPLANATION_LOG {
        int id PK
        string user_id FK
        string disease
        string model_name
        int predicted_class
        float risk_score
        string risk_level
        json thresholds
        string threshold_band
        json major_risk_factors
        json validation_warnings
        json raw_inputs
        json transformed_features
        json top_factors
        json calibration_context
        text explanation_text
        string source "api|terminal_explain_raw"
        timestamp created_at
    }

    DIET_PLAN {
        int id PK
        string user_id FK
        string disease
        string plan_type
        timestamp created_at
    }

    MEAL_LOG {
        int id PK
        string user_id FK
        string meal_name
        date date_logged
        timestamp created_at
    }

    ALERT {
        int id PK
        string user_id FK
        string alert_type
        string severity
        timestamp created_at
    }
```

## 4. Model Performance Comparison

```mermaid
graph LR
    subgraph "CKD Model"
        CKD["RandomForest<br/>F1: 0.9868<br/>ROC-AUC: 1.0000<br/>150 trees<br/>10 features"]
    end

    subgraph "Hypertension Model"
        HTN["RandomForest<br/>F1: 0.8764<br/>ROC-AUC: 0.9451<br/>150 trees<br/>13 features"]
    end

    subgraph "Diabetes Model"
        DM["RandomForest<br/>F1: 0.7175<br/>ROC-AUC: 0.8096<br/>150 trees<br/>10 features"]
    end

    CKD -->|Load at runtime| PRED["🤖 Prediction Engine"]
    HTN -->|Load at runtime| PRED
    DM -->|Load at runtime| PRED

    PRED -->|<50ms latency| SCORE["Risk Score<br/>0.0 - 1.0"]
```

## 5. Risk Scoring & Interpretation

```mermaid
graph TD
    SCORE["Predicted Risk Score<br/>(0.0 - 1.0)"]
    
    SCORE -->|< 0.35| LOW["🟢 LOW RISK"]
    SCORE -->|0.35 - 0.9| MOD["🟡 MODERATE RISK"]
    SCORE -->|> 0.9| HIGH["🔴 HIGH RISK"]
    
    LOW -->|Top factors| EXP1["Explanation:<br/>Your risk is low<br/>Continue preventive care"]
    MOD -->|Top factors| EXP2["Explanation:<br/>Your risk is moderate<br/>Implement diet changes"]
    HIGH -->|Top factors| EXP3["Explanation:<br/>Your risk is high<br/>Immediate clinic visit"]
    
    EXP1 -->|With warnings| OUTPUT["RiskExplanationRawResponse<br/>• risk_score<br/>• risk_level<br/>• top_factors<br/>• validation_warnings<br/>• explanation_text<br/>• calibration_context"]
    EXP2 -->|With warnings| OUTPUT
    EXP3 -->|With warnings| OUTPUT
    
    OUTPUT -->|Persist| LOG["risk_explanation_log<br/>(SQLite audit trail)"]
```

## 6. Integration Architecture (Current & Future)

```mermaid
graph TB
    subgraph "Current Implementation"
        API["✓ FastAPI REST API"]
        TERM["✓ Terminal/CLI"]
        FILE["✓ File System<br/>CSV, JSON"]
        DB["✓ SQLite<br/>Local DB"]
    end

    subgraph "Future Integrations"
        EHR["🔜 EHR Systems<br/>(Epic, Cerner)"]
        HL7["🔜 HL7/FHIR<br/>Standard"]
        CLOUD["🔜 Cloud Platform<br/>(AWS, Azure)"]
        DASH["🔜 Analytics Dashboard<br/>(BI tools)"]
    end

    API --> EHR
    API --> HL7
    DB --> CLOUD
    DB --> DASH
    TERM --> DB
    FILE --> API

    style API fill:#90EE90
    style TERM fill:#90EE90
    style FILE fill:#90EE90
    style DB fill:#90EE90
    style EHR fill:#FFB6C6
    style HL7 fill:#FFB6C6
    style CLOUD fill:#FFB6C6
    style DASH fill:#FFB6C6
```

## 7. Deployment Architecture

```mermaid
graph LR
    subgraph "On-Premise Deployment"
        direction TB
        SERVER["Linux/Windows Server<br/>Python 3.13"]
        FASTAPI["FastAPI App<br/>(Port 8000)"]
        SQLITE["SQLite Database<br/>app_data.db"]
        MODELS["ML Models<br/>(joblib)"]
        
        SERVER --> FASTAPI
        FASTAPI --> SQLITE
        FASTAPI --> MODELS
    end

    subgraph "Client Access"
        direction LR
        CLI["Terminal/CLI"]
        WEB["Web Browser"]
        MOBILE["Mobile App"]
    end

    CLI -->|HTTP| FASTAPI
    WEB -->|HTTP| FASTAPI
    MOBILE -->|HTTP| FASTAPI

    subgraph "Future: Cloud Deployment"
        CLOUD["AWS/Azure<br/>Container (Docker)<br/>Kubernetes Orchestration"]
    end

    FASTAPI -.->|Future| CLOUD

    style SERVER fill:#E3F2FD
    style FASTAPI fill:#E3F2FD
    style SQLITE fill:#E3F2FD
    style MODELS fill:#E3F2FD
    style CLOUD fill:#FFE0B2
```

## 8. Quality & Governance

```mermaid
graph TB
    subgraph "Model Governance"
        MODELS["ML Models<br/>(artifacts/)"]
        VERSION["Version Control<br/>(joblib hash ID)"]
        FEATURES["Feature Lists<br/>({disease}_features.json)"]
        THRESHOLDS["Risk Thresholds<br/>(risk_thresholds_and_factors.json)"]
    end

    subgraph "Audit & Compliance"
        AUDIT["Audit Logging<br/>(risk_explanation_log)"]
        TIMESTAMP["Timestamp<br/>All Predictions"]
        SOURCE["Track Source<br/>(API/Terminal)"]
        PAYLOAD["Full Payload<br/>Persistence"]
    end

    subgraph "Quality Metrics"
        PERF["Performance<br/>(<50ms latency)"]
        ACC["Accuracy<br/>(F1, ROC-AUC)"]
        UP["Uptime<br/>(99.9% target)"]
        SEC["Security<br/>(HIPAA-ready)"]
    end

    MODELS --> VERSION
    MODELS --> FEATURES
    MODELS --> THRESHOLDS

    AUDIT --> TIMESTAMP
    AUDIT --> SOURCE
    AUDIT --> PAYLOAD

    VERSION --> PERF
    FEATURES --> ACC
    PAYLOAD --> UP
    TIMESTAMP --> SEC

    style MODELS fill:#C8E6C9
    style AUDIT fill:#BBDEFB
    style PERF fill:#F8BBD0
    style ACC fill:#F8BBD0
    style UP fill:#F8BBD0
    style SEC fill:#F8BBD0
```

---

**Note:** These diagrams complement the detailed SYSTEM_DESIGN_OVERVIEW.md document. Use them for:
- PowerPoint slides (copy-paste Mermaid output or export as PNG/SVG)
- Documentation
- Stakeholder presentations
- Architecture review meetings
