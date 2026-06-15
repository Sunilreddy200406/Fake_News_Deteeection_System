# Hybrid System Flow Diagram

```mermaid
flowchart TD
    A[User submits news text + optional source URL] --> B[Flask API /analyze]
    B --> C[Preprocess text]
    C --> D[Extract keywords + entities]
    D --> E[Fetch related articles from official portals via Google News RSS]
    E --> F[Compute similarity with official articles]
    F --> G{Similarity >= Threshold OR Trusted Source URL?}
    G -->|Yes| H[Real News (Verified Official Source)]
    G -->|No| I[TF-IDF Vectorization]
    I --> J[ML models: Logistic Regression / Naive Bayes]
    J --> K[Decision Engine]
    K --> L[Fake News / Real (Unverified) / Suspicious]
    H --> M[Return explainable JSON response]
    L --> M
    M --> N[Display result in frontend]
```

