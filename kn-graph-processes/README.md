1) Parse BPMN --> Graph tensors:
  - Nodes: tasks, gateways, events.
  - Edges: sequence flows.
  - Features: one-hot type, normalized numeric attributes (if available).

2) Run message-passing kernel:
- Each node updates its vector by aggregating neighbor features (mean + linear + ReLU).
- Repeat K iterations --> contextual embeddings.

3) Pool to process-level embedding:
- Mean pooling across nodes in each process.

4) Visualisation:
- Use cosine similarity, PCA, or t-SNE to find clusters of similar processes or anomalies.

