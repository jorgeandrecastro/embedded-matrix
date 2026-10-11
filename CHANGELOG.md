# Changelog

## [0.2.0] - 2026-10-11

### Corrections
- **Seuil de singularité relatif** : `inv` n'utilise plus `|det| < 1e-7` (dépendant de l'échelle). Une matrice est singulière si `|det| <= 8·EPSILON × (somme des |termes| du déterminant)`. `1e-4 · I` est désormais inversible ; les matrices à grands coefficients dont le déterminant n'est que du bruit d'arrondi sont rejetées ; la matrice nulle est singulière.
- `inv` ne renvoie plus de matrice de NaN quand l'entrée contient NaN ou ±∞.
- Documentation de `Matrix3x3::det` : développement par cofacteurs (et non « règle de Sarrus »).

### Changements (cassants)
- `MatrixError` devient `#[non_exhaustive]` et gagne la variante `NonFiniteValue` : ajoutez un bras `Err(_)` à vos `match`.
- Version minimale de Rust : 1.85 (édition 2024).

### Ajouts
- `MatrixError` implémente `Display` et `core::error::Error`.
- Produit matrice × vecteur : `mul_vec([f32; N])` et `matrice * [f32; N]`.
- `scalaire * matrice`, `+=`, `-=`, `*=` (matrice et scalaire).
- Accès par indice `m[(ligne, colonne)]`.
- `From<[[f32; N]; N]>` et conversion inverse, `Default` (matrice nulle), `from_diagonal`, `approx_eq`.

### Améliorations
- `Matrix3x3::inv` réutilise les cofacteurs de la première ligne pour calculer le déterminant.
- Code découpé en modules (`error`, `matrix2x2`, `matrix3x3`) ; tests déplacés dans `tests/matrix.rs` (valeurs connues, robustesse, propriétés sur matrices pseudo-aléatoires).
- Les méthodes `add`, `sub`, `mul` restent disponibles pour la compatibilité.

## [0.1.0]

- Première version : `Matrix2x2` et `Matrix3x3` (`new`, `identity`, `zero`, `get`, `as_array`, `add`, `sub`, `mul`, `scale`, `transpose`, `det`, `trace`, `inv`), opérateurs `+`, `-`, `*`, `-a`.