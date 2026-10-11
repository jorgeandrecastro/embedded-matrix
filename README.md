# embedded-matrix

Matrices 2×2 et 3×3 en `f32` pour systèmes embarqués `no_std`.

[![Crates.io](https://img.shields.io/crates/v/embedded-matrix.svg)](https://crates.io/crates/embedded-matrix)
[![License](https://img.shields.io/crates/l/embedded-matrix.svg)](LICENSE)
[![Docs.rs](https://docs.rs/embedded-matrix/badge.svg)](https://docs.rs/embedded-matrix/)

La version 0.3 ajoute `solve` (résolution de `A·x = b`) et, derrière la feature optionnelle `trig`, les constructeurs de rotation. La 0.2 avait corrigé le seuil de détection des matrices singulières (désormais **relatif à l'échelle**), ajouté `NonFiniteValue` aux erreurs, le produit matrice × vecteur et plusieurs opérateurs. Voir le [CHANGELOG](CHANGELOG.md).

## Points forts

- **Zéro dépendance par défaut** : pas de libm, pas de micromath, pas de bibliothèque C, pas d'allocation.
- **Sécurité** : `#![forbid(unsafe_code)]` et inversion via `Result<_, MatrixError>`.
- **Portabilité** : fonctionne sur Cortex-M0+, M4F, M33, RISC-V et toute cible `no_std`.
- **Opérateurs natifs** : `a * b`, `a + b`, `a - b`, `a * 2.0`, `2.0 * a`, `-a`, `a * [x, y, z]`, `+=`, `-=`, `*=`.
- **Inversion robuste** : seuil de singularité relatif, rejet explicite de NaN et ±∞.

## Utilisation

```toml
[dependencies]
embedded-matrix = "0.3"
```

Version minimale de Rust : **1.85** (édition 2024).

### Feature optionnelle `trig`

```toml
[dependencies]
embedded-matrix = { version = "0.3", features = ["trig"] }
```

Active la dépendance à [`embedded-trig-f32`](https://crates.io/crates/embedded-trig-f32) et les constructeurs de rotation (angles en radians, règle de la main droite). **Sans cette feature, la crate n'a aucune dépendance.**

```rust
use embedded_matrix::{Matrix2x2, Matrix3x3};

let r = Matrix2x2::rotation(0.5);
let p = r * [1.0, 0.0];

let rz = Matrix3x3::rotation_z(0.5);
let q = rz * [1.0, 0.0, 5.0];   // z reste inchangé
```

Pour une rotation, l'inverse est la transposée : `r.transpose()` suffit, inutile d'appeler `inv()`.

## Opérations disponibles

| Méthode              | Opérateur       | Description                                             |
|----------------------|-----------------|---------------------------------------------------------|
| `new(data)`          | `From<[[f32; N]; N]>` | Construit depuis un tableau row-major             |
| `identity()`         | —               | Matrice identité                                        |
| `zero()`             | `Default`       | Matrice nulle                                           |
| `from_diagonal(d)`   | —               | Matrice diagonale                                       |
| `add(&m)`            | `a + b`, `+=`   | Addition terme à terme                                  |
| `sub(&m)`            | `a - b`, `-=`   | Soustraction terme à terme                              |
| `mul(&m)`            | `a * b`, `*=`   | Produit matriciel                                       |
| `scale(f32)`         | `a * 2.0`, `2.0 * a`, `*= 2.0` | Multiplication par un scalaire           |
| `mul_vec([f32; N])`  | `a * v`         | Produit matrice × vecteur colonne                       |
| `solve([f32; N])`    | —               | Résout `A·x = b` → `Result<[f32; N], MatrixError>`      |
| `rotation(angle)` ¹  | —               | Rotation 2D (`Matrix2x2`)                               |
| `rotation_x/y/z(angle)` ¹ | —          | Rotations 3D (`Matrix3x3`)                              |
| —                    | `-a`            | Négation terme à terme                                  |
| `transpose()`        | —               | Transposée                                              |
| `det()`              | —               | Déterminant                                             |
| `trace()`            | —               | Trace (somme de la diagonale)                           |
| `inv()`              | —               | Inverse → `Result<M, MatrixError>`                      |
| `get(row, col)`      | `m[(row, col)]` | Accès à un élément (panique si hors limites)            |
| `as_array()`         | `Into<[[f32; N]; N]>` | Accès aux données row-major                       |
| `approx_eq(&m, tol)` | —               | Égalité à une tolérance absolue près                    |

¹ Disponible avec la feature `trig` (voir plus haut).

### Erreurs

| Variante                       | Cas                                                              |
|--------------------------------|------------------------------------------------------------------|
| `MatrixError::SingularMatrix`  | Déterminant nul ou négligeable devant l'erreur d'arrondi         |
| `MatrixError::NonFiniteValue`  | La matrice contient NaN ou ±∞, ou le calcul a débordé            |

`MatrixError` est `#[non_exhaustive]` : prévoyez un bras `Err(_)` dans vos `match`. Il implémente `Display` et `core::error::Error`.

## Exemples

### Syntaxe opérateurs

```rust
use embedded_matrix::Matrix2x2;

let a = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
let b = Matrix2x2::identity();

let c = a * b;      // produit matriciel
let d = a + b;      // addition terme à terme
let e = a - b;      // soustraction terme à terme
let f = a * 2.0;    // scalaire
let g = 2.0 * a;    // scalaire à gauche
let h = -a;         // négation
let v = a * [1.0, 1.0]; // matrice × vecteur → [3.0, 7.0]

let mut m = a;
m += b;
m *= 0.5;
```

### Utilisation complète

```rust
use embedded_matrix::{Matrix2x2, Matrix3x3, MatrixError};

// Matrice 2×2
let a = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
let det = a.det();   // -2.0

match a.inv() {
    Ok(inv) => { let _ = a * inv; /* ≈ identité */ }
    Err(MatrixError::SingularMatrix) => { /* matrice singulière */ }
    Err(_) => { /* NaN / infini */ }
}

// Matrice 3×3
let r = Matrix3x3::new([
    [ 2.0, -1.0,  0.0],
    [-1.0,  2.0, -1.0],
    [ 0.0, -1.0,  2.0],
]);
let inv = r.inv().unwrap();
let trace = r.trace();  // 6.0
let det   = r.det();    // 4.0
let x = r * [1.0, 0.0, -1.0]; // matrice × vecteur
```

### Résoudre un système `A·x = b`

```rust
use embedded_matrix::{Matrix3x3, MatrixError};

let a = Matrix3x3::new([
    [1.0, 2.0, 3.0],
    [0.0, 1.0, 4.0],
    [5.0, 6.0, 0.0],
]);

match a.solve([1.0, 2.0, 3.0]) {
    Ok(x) => { /* x ≈ [27.0, -22.0, 6.0] */ let _ = x; }
    Err(MatrixError::SingularMatrix) => { /* pas de solution unique */ }
    Err(_) => { /* NaN / infini */ }
}
```

`solve` applique le même critère de singularité que `inv()`. En 2×2 il utilise la formule de Cramer directement ; en 3×3 il calcule `A⁻¹·b` (même coût que `inv`). Pour plusieurs seconds membres avec la même matrice, appelez `inv()` une seule fois.

### Exemple embarqué : norme de Frobenius avec embedded-f32-sqrt (RP2350)

```rust
use embedded_matrix::Matrix3x3;
use embedded_f32_sqrt::sqrt;

// Norme de Frobenius : sqrt(Σ aij²)
fn frobenius_norm(m: &Matrix3x3) -> f32 {
    let mut sum = 0.0f32;
    for row in m.as_array() {
        for &v in row {
            sum += v * v;
        }
    }
    sqrt(sum).unwrap_or(0.0)
}
```

## Algorithmes

- **`Matrix2x2`** : inversion par formule directe, `1/det * [[d, -b], [-c, a]]`.
- **`Matrix3x3`** : déterminant par développement selon la première ligne (cofacteurs) ; inversion par la méthode des cofacteurs (matrice adjointe divisée par le déterminant), en réutilisant les cofacteurs de la première ligne pour le déterminant.

### Détection des matrices singulières

Le seuil est **relatif** : une matrice est déclarée singulière si

```
|det| <= 8 · f32::EPSILON · B
```

où `B` est la somme des valeurs absolues des termes qui composent le déterminant (`|ad| + |bc|` en 2×2, les six produits du développement en 3×3). `B` donne l'échelle de l'erreur d'arrondi attendue sur `det`.

Conséquences :

- `1e-4 · I` reste inversible (l'ancien seuil absolu `|det| < 1e-7` la rejetait à tort).
- Une matrice à grands coefficients dont le déterminant n'est que du bruit d'arrondi est rejetée.
- La matrice nulle est singulière.

Ce test ne mesure pas le conditionnement : une matrice inversible mais très mal conditionnée donnera un inverse peu précis.

## Tests

```bash
cargo test                    # tests et doc-tests, sans dépendance
cargo test --features trig    # + rotations
```

La suite (`tests/matrix.rs`) vérifie les valeurs connues, la robustesse (échelles, NaN, ∞, matrice nulle) et des propriétés sur des matrices pseudo-aléatoires déterministes : `A·A⁻¹ ≈ I`, `det(AB) = det(A)·det(B)`, `(AB)ᵀ = BᵀAᵀ`, `solve`, et (avec `trig`) les propriétés des rotations : déterminant 1, orthogonalité, composition, règle de la main droite.

## Pistes futures (non implémentées)

Volontairement hors de l'API actuelle, pour garder une crate simple et prévisible :

- **Taille quelconque** : `Matrix<const R: usize, const C: usize>` sur la pile (alias `Matrix2x2 = Matrix<2, 2>`), avec produit de matrices non carrées vérifié à la compilation.
- **Décomposition LU** avec pivot partiel pour `det`, `inv` et `solve` en taille n.
- **Cholesky** pour les matrices symétriques définies positives (matrices de covariance, filtres de Kalman).
- **Type `Vector<N>`** : produit scalaire, produit vectoriel 3D, norme (via `embedded-f32-sqrt`).
- **`#[must_use]`** sur les opérations arithmétiques.
- **Benchmarks en cycles** (compteur DWT sur Cortex-M) et CI de compilation sur cible embarquée.

## Licence

GPL-2.0-or-later

Copyright (C) 2026 Jorge Andre Castro.