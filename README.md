# nyas : A simple Vector Database
Nyas is a lightweight and efficient vector database engine designed for similarity search of embeddings.
It supports multiple distance metrics (L2, Cosine, Dot), hybrid CPU/GPU execution, and flexible storage backends.

---

## References

- *“DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node”* (NeurIPS 2019)
  https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf

- *“FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search”* (arXiv 2021)
  https://arxiv.org/abs/2105.09613

These papers influenced Nyas’s design, particularly in indexing structures, graph search, and efficient query traversal.
