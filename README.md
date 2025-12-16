Independent Study: LLM Behavioral Fingerprinting via Correlated Errors

This repository contains the code, datasets, and analysis for the independent study on black-box LLM fingerprinting using error patterns from the HellaSwag dataset. Conducted by Arnav Desai (117543661) under Prof. Amir Rahmati in Fall 2025, the project demonstrates model family attribution via behavioral features like confusion matrices and confidence margins.
Project Overview

LLM fingerprinting identifies underlying models through observable outputs without internal access. This work leverages HellaSwag (~10k multiple-choice commonsense examples) to capture correlated errors, revealing intra-family consistency (e.g., Qwen Jaccard index 0.513) versus inter-family divergence (0.306 vs. Stable Zephyr). ErrorTrace features—correctness, entropy, error-conditioned confidence—enable fingerprints for Logistic Regression classifiers.
Key Contributions

    Error overlap quantification validating family-level separability.

    Probe selection via inter-model disagreement for efficiency.

    Hybrid replication of LLMmap, Behavioral Fingerprinting, and Invisible Traces techniques.

Core Methodology

    Probing: Query models on HellaSwag in multiple-choice format, record full probability distributions mei∈R4mei∈R4.

    Feature Extraction: Compute per-probe vectors fm,i(mei,yi)fm,i(mei,yi) including margins and entropy; aggregate to fingerprint FmFm.

    Selection: Rank examples by variance in features for high-disagreement subsets (e.g., top 3000).

    Attribution: Train classifiers for family (Qwen vs. Stable) or all-model identification.
