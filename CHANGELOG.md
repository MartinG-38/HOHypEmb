# Changelog

## 0.1.0 — unreleased

- HHO_3 embedding (`HHOEmbedder`, layer-2 modes `input` / `maximal` / `all_simplices`) and the
  classical embedding, reproducing MLNHypEmb to numerical precision (golden tests).
- Six baselines (weighted clique, star expansion, Zhou hypergraph, three Euclidean) and a registry.
  The weighted clique adds the shared simplex counts to the edge weights after the RA preweighting
  and keeps the classical radii, so it differs from the classical embedding by that one change.
- Markers, evaluation statistics (incl. partial Spearman), Poincaré-disk plotting.
- Datasets: planted-surface synthetic graphs, cora co-authorship (AHORN, Zenodo), E. coli
  (STRING + Complex Portal), with cached, checksummed downloads.
- Four tutorial notebooks in `examples/` (quickstart, cora, E. coli, synthetic markers) that use only the
  public API and reproduce the main results of the paper.
- Flat package layout: copy the `hohypemb/` folder into your project and import it, no installation needed.
