### Retrieval Failure Analysis

During manual testing, the single-word query `heterogen` retrieved the
concept `q002 — Konfusionsmatrix`, although the expected result was
`q003 — Clusteranalyse`.

| Query | Expected | Retrieved | Result |
|---|---|---|---|
| `heterogen` | `q003 — Clusteranalyse` | `q002 — Konfusionsmatrix` | Incorrect |

The word `heterogen` appears explicitly in the Clusteranalyse document:

> Die Cluster sollten untereinander maximal heterogen und innerhalb maximal homogen sein.

The error occurred during retrieval, not generation. The LLM received the
Konfusionsmatrix document as context and generated an answer based on that
incorrectly retrieved document.

This test demonstrates a limitation of dense vector search for short,
single-word vocabulary queries. Semantic embeddings may interpret an
isolated word differently from an exact lexical match.

Possible improvements include:

- adding short vocabulary queries to the retrieval evaluation dataset;
- combining lexical and vector retrieval;
- using exact phrase matching before vector search;
- evaluating a hybrid search approach;
- applying document re-ranking after retrieval.