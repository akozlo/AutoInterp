"""Curated seed papers for AI interpretability citation graph.

Each entry has:
  - id: Semantic Scholar paper ID, or ARXIV:/DOI: prefixed identifier
  - title: Paper title (used as fallback for title-match lookup)
  - group: Category for annotation
"""

SEED_PAPERS = [
    # --- Distill Circuits thread ---
    {
        "id": "DOI:10.23915/distill.00024.001",
        "title": "Zoom In: An Introduction to Circuits",
        "group": "distill-circuits",
    },
    {
        "id": "DOI:10.23915/distill.00024.003",
        "title": "Curve Detectors",
        "group": "distill-circuits",
    },
    {
        "id": "DOI:10.23915/distill.00024.004",
        "title": "Naturally Occurring Equivariance in Neural Networks",
        "group": "distill-circuits",
    },
    {
        "id": "DOI:10.23915/distill.00024.005",
        "title": "An Overview of Early Vision in InceptionV1",
        "group": "distill-circuits",
    },
    # --- Anthropic interpretability ---
    {
        "id": "ARXIV:2209.11895",
        "title": "In-context Learning and Induction Heads",
        "group": "anthropic",
    },
    {
        "id": "ARXIV:2209.10652",
        "title": "Toy Models of Superposition",
        "group": "anthropic",
    },
    {
        "id": "ARXIV:2406.04093",
        "title": "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet",
        "group": "anthropic",
    },
    # --- Key academic papers ---
    {
        "id": "ARXIV:2202.05262",
        "title": "Locating and Editing Factual Associations in GPT",
        "group": "academic",
    },
    {
        "id": None,  # arXiv ID mismatch; use title match
        "title": "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small",
        "group": "academic",
    },
    {
        "id": "ARXIV:2301.05217",
        "title": "Progress measures for grokking via mechanistic interpretability",
        "group": "academic",
    },
    {
        "id": "ARXIV:2210.01892",
        "title": "Polysemanticity and Capacity in Neural Networks",
        "group": "academic",
    },
    {
        "id": "ARXIV:2210.13382",
        "title": "Othello-GPT",
        "group": "academic",
    },
    {
        "id": "ARXIV:2304.14997",
        "title": "Towards Automated Circuit Discovery for Mechanistic Interpretability",
        "group": "academic",
    },
    {
        "id": "ARXIV:2305.01610",
        "title": "Language Models Can Explain Neurons in Language Models",
        "group": "academic",
    },
    {
        "id": "ARXIV:2310.10348",
        "title": "Representation Engineering: A Top-Down Approach to AI Transparency",
        "group": "academic",
    },
    {
        "id": "ARXIV:2312.06681",
        "title": "Steering Llama 2 via Contrastive Activation Addition",
        "group": "academic",
    },
    {
        "id": "ARXIV:2507.21509",
        "title": "Persona Vectors: Monitoring and Controlling Character Traits in Language Models",
        "group": "academic",
    },
    {
        "id": "ARXIV:2310.06824",
        "title": "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets",
        "group": "academic",
    },
    {
        "id": "ARXIV:2502.17424",
        "title": "Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs",
        "group": "academic",
    },
    {
        "id": "ARXIV:2501.11120",
        "title": "Tell me about yourself: LLMs are aware of their learned behaviors",
        "group": "academic",
    },
]
