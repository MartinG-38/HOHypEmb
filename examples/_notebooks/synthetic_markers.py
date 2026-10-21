"""Cells of 04_synthetic_markers.ipynb: the area markers on two models."""
from . import code, md

CELLS = [
    md(r"""
    # Reading the area markers on synthetic graphs

    Two random-graph models (Holme–Kim and Erdős–Rényi), each with about half of its triangles filled. For each one we fit the
    classical and the HHO embedding, compute the area markers of `hohypemb.markers`. This
    notebook reproduces results of the paper.
    """),
    code(r"""
    import sys

    import matplotlib.pyplot as plt
    import pandas as pd

    sys.path.insert(0, "..")  # finds hohypemb/ in the repository; not needed if you copied or installed it
    import hohypemb as hh
    from hohypemb.datasets.synthetic import planted_surfaces

    MODELS = ["holme_kim", "erdos_renyi"]
    GAMMA = 0.5
    complexes = {m: planted_surfaces(m, p_surface=50).giant_component() for m in MODELS}
    pd.DataFrame({m: {"nodes": sc.n_nodes, "edges": sc.one_skeleton.number_of_edges(),
                      "filled triangles": len(sc.filled_triangles()),
                      "empty triangles": len(sc.empty_triangles())}
                  for m, sc in complexes.items()}).T
    """),
    md(r"""
    ## 1. The classical and the HHO embedding

    Holme–Kim, μ = 15, γ = 0.5: the classical embedding sees only the graph, HHO also sees the filled
    triangles (shaded). The two fits are reused in section 2.
    """),
    code(r"""
    MU = 15.0
    fits = {}
    sc = complexes["holme_kim"]
    params = hh.presets.EMBEDDING_PARAMS["synthetic_holme_kim"]
    cl = hh.ClassicalEmbedder(**params).fit(sc, seed=0)
    hho = hh.HHOEmbedder(mu=MU, gamma=GAMMA, **params).fit(sc, seed=0)
    fits["holme_kim"] = (cl, hho)
    fig = hh.plotting.plot_comparison(sc, cl, hho)
    plt.show()
    """),
    md(r"""
    ## 2. The area markers, on every model

    Each model has its own preset (`beta` matched to its degree scale); μ = 15, γ = 0.5.

    | column | how to read it |
    |---|---|
    | `area ratio` | median area HHO / classical of the filled triangles (below 1: HHO shrinks them) |
    | `shrunk` | fraction of filled triangles that shrink |
    | `filled vs empty` | under HHO, median empty area / median filled area |
    """),
    code(r"""
    rows = {}
    for model, sc in complexes.items():
        params = hh.presets.EMBEDDING_PARAMS[f"synthetic_{model}"]
        if model not in fits:  # Holme-Kim was fitted in section 1
            cl = hh.ClassicalEmbedder(**params).fit(sc, seed=0)
            hho = hh.HHOEmbedder(mu=MU, gamma=GAMMA, **params).fit(sc, seed=0)
            fits[model] = (cl, hho)
        cl, hho = fits[model]
        area = hh.markers.area_ratio(sc, cl, hho)
        surf = hh.markers.surface_vs_empty_areas(sc, hho)
        rows[model] = {
            "area ratio": area.median_ratio,
            "shrunk": area.frac_below_1,
            "filled vs empty": surf.fold,
        }
    markers_table = pd.DataFrame(rows).T
    markers_table.round(3)
    """),
    md(r"""
    *What to expect:* the median area ratio is below 1 on both models (0.41 on Holme–Kim, 0.60 on
    Erdős–Rényi), and 77% and 66% of the filled triangles shrink. Under HHO the empty triangles are
    larger than the filled ones on both models (fold 8.3 on Holme–Kim, 1.3 on Erdős–Rényi).
    """),
    md(r"""
    ## 3. The synthesis figure

    Holme–Kim: (A) area ratio of the filled triangles, (B) filled vs empty triangles under HHO, (C)
    embedding distance against hop distance in both embeddings.
    """),
    code(r"""
    sc = complexes["holme_kim"]
    cl, hho = fits["holme_kim"]
    fig = hh.plotting.plot_synthesis(hh.markers.area_ratio(sc, cl, hho),
                                     hh.markers.surface_vs_empty_areas(sc, hho),
                                     hh.markers.global_distortion(sc, cl),
                                     hh.markers.global_distortion(sc, hho))
    plt.show()
    """),
    md(r"""
    *What to expect:* (A) the median area ratio is 0.41, and 77% of the 87 filled triangles are
    smaller under HHO. (B) Under HHO the areas of the filled triangles are spread over smaller values
    than those of the 91 empty triangles. (C) The distortion ρ is -0.02 for the classical embedding
    and 0.40 for HHO, and the embedding distance at a given hop distance differs between the two.
    """),
]
