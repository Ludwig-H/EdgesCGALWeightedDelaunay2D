# EdgesCGALWeightedDelaunay2D

Calcule les **arêtes** du 1‑squelette de la **Delaunay pondérée (regular triangulation) 2D** à partir d’un `.npy` `points (N,2)` et d’un `.npy` `weights (N,)`. Sortie : `.npy` `(M,2)` `uint64` avec paires **triées** `(i<j)`.

## Dépendances
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libcgal-dev
```

## Build
```bash
cmake -S . -B build
cmake --build build -j
```

## Usage
```bash
./build/EdgesCGALWeightedDelaunay2D points.npy weights.npy out_edges.npy
```
- `points.npy` : `float32`/`float64`, shape `(N,2)`, C‑contigu. Endianness `<`, `=`, `>` supportées (byteswap auto si `>`).
- `weights.npy` : `float32`/`float64`, shape `(N,)` ou `(N,1)`. **Poids de puissance** (souvent `rayon^2`).

## Implémentation
- Traits 2D spécialisés: `Regular_triangulation_euclidean_traits_2<K>` et `Regular_triangulation_2`.
- Vertex avec info: `Triangulation_vertex_base_with_info_2<uint64_t, Gt, Regular_triangulation_vertex_base_2<Gt>>`.
- Arêtes via `finite_edges` puis tri + unique.