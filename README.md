# embedded-matrix

Matrices 2×2 et 3×3 en `f32` pour systèmes embarqués `no_std`.

[![Crates.io](https://img.shields.io/crates/v/embedded-matrix.svg)](https://crates.io/crates/embedded-matrix)
[![License](https://img.shields.io/crates/l/embedded-matrix.svg)](LICENSE)
[![Docs.rs](https://docs.rs/embedded-matrix/badge.svg)](https://docs.rs/embedded-matrix/)

## Points forts

- **Zéro dépendance** — pas de libm, pas de micromath, pas de bibliothèque C.
- **Sécurité maximale** — `#![forbid(unsafe_code)]` et gestion robuste via `Result`.
- **Portabilité totale** — fonctionne sur Cortex-M0+, M4F, M33, RISC-V et tout autre `no_std`.
- **Opérateurs natifs** — `a * b`, `a + b`, `a - b`, `a * 2.0`, `-a` grâce aux traits `core::ops`.

## Utilisation

```toml
[dependencies]
embedded-matrix = "0.1.0"
```

## Opérations disponibles

| Méthode | Opérateur | Description |
|---|---|---|
| `new(data)` | — | Construit depuis un tableau row-major |
| `identity()` | — | Matrice identité |
| `zero()` | — | Matrice nulle |
| `add(&m)` | `a + b` | Addition terme à terme |
| `sub(&m)` | `a - b` | Soustraction terme à terme |
| `mul(&m)` | `a * b` | Produit matriciel |
| `scale(f32)` | `a * 2.0` | Multiplication par un scalaire |
| `transpose()` | — | Transposée |
| `det()` | — | Déterminant |
| `trace()` | — | Trace (somme de la diagonale) |
| `inv()` | — | Inverse → `Result<M, MatrixError>` |
| `get(row, col)` | — | Accès à un élément |

## Exemples

### Syntaxe opérateurs

```rust
use embedded_matrix::{Matrix2x2, Matrix3x3};

let a = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
let b = Matrix2x2::identity();

let c = a * b;      // produit matriciel
let d = a + b;      // addition terme à terme
let e = a - b;      // soustraction terme à terme
let f = a * 2.0;    // scalaire
let g = -a;         // négation
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
```

### Exemple embarqué norme de Frobenius avec embedded-f32-sqrt sur RP2350

```rust
use embedded_matrix::Matrix3x3;
use embedded_f32_sqrt::sqrt;

// Norme de Frobenius : sqrt(Σ aij²)
fn frobenius_norm(m: &Matrix3x3) -> f32 {
    let arr = m.as_array();
    let mut sum = 0.0f32;
    for row in arr {
        for &v in row {
            sum += v * v;
        }
    }
    sqrt(sum).unwrap_or(0.0)
}
```

## Algorithmes

`Matrix2x2` inversion par formule directe : `1/det * [[d, -b], [-c, a]]`.

`Matrix3x3`  déterminant par la règle de Sarrus ; inversion par la méthode des cofacteurs (matrice adjointe divisée par le déterminant).

Le seuil de singularité est `|det| < 1e-7`, choisi pour correspondre à la précision du type `f32`.

## Licence

GPL-2.0-or-later

Copyright (C) 2026 Jorge Andre Castro.