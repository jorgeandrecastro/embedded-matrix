// Copyright (C) 2026 Jorge Andre Castro
//
// Ce programme est un logiciel libre : vous pouvez le redistribuer et/ou le modifier
// selon les termes de la Licence Publique Générale GNU telle que publiée par la
// Free Software Foundation, soit la version 2 de la licence, soit (à votre convention)
// n'importe quelle version ultérieure.

//! # embedded-matrix
//!
//! Matrices 2×2 et 3×3 en `f32` pour systèmes embarqués `no_std`.
//!
//! Sans dépendance, sans `unsafe`, sans allocation.
//!
//! ```rust
//! use embedded_matrix::{Matrix2x2, Matrix3x3, MatrixError};
//!
//! // Produit matriciel avec l'opérateur *
//! let a = Matrix3x3::new([
//!     [1.0, 2.0, 3.0],
//!     [4.0, 5.0, 6.0],
//!     [7.0, 8.0, 9.0],
//! ]);
//! let b = Matrix3x3::identity();
//! assert_eq!(a * b, a);
//!
//! // Addition, soustraction, scalaire
//! assert_eq!((a + b) - b, a);
//! let _double = a * 2.0;
//!
//! // Matrice × vecteur
//! let v = Matrix2x2::identity() * [1.0, 2.0];
//! assert_eq!(v, [1.0, 2.0]);
//!
//! // Inversion (Err si singulière ou non finie)
//! match Matrix2x2::new([[2.0, 0.0], [0.0, 4.0]]).inv() {
//!     Ok(inv) => assert_eq!(inv, Matrix2x2::new([[0.5, 0.0], [0.0, 0.25]])),
//!     Err(MatrixError::SingularMatrix) => { /* gérer */ }
//!     Err(e) => panic!("{e}"),
//! }
//! ```
//!
//! ## Résolution de systèmes
//!
//! `solve(b)` résout `A·x = b` et renvoie directement `x` :
//!
//! ```rust
//! use embedded_matrix::Matrix2x2;
//!
//! let a = Matrix2x2::new([[2.0, 1.0], [1.0, 3.0]]);
//! let x = a.solve([3.0, 5.0]).unwrap(); // x = [0.8, 1.4]
//! assert!((x[0] - 0.8).abs() < 1e-6);
//! ```
//!
//! ## Feature optionnelle : `trig`
//!
//! ```toml
//! embedded-matrix = { version = "0.3", features = ["trig"] }
//! ```
//!
//! Active la dépendance à `embedded-trig-f32` et ajoute les constructeurs de
//! rotation : `Matrix2x2::rotation(angle)`, `Matrix3x3::rotation_x / _y / _z(angle)`
//! (angles en radians, règle de la main droite). Sans la feature, la crate
//! n'a **aucune dépendance**.
//!
//! ## Pistes futures (non implémentées)
//!
//! Ces idées ne font pas partie de l'API actuelle ; elles sont notées ici
//! pour la suite :
//!
//! - **Taille quelconque** : `Matrix<const R: usize, const C: usize>` sur la
//!   pile (alias `Matrix2x2 = Matrix<2, 2>`), produit de matrices non carrées.
//! - **Décomposition LU** avec pivot partiel pour `det`, `inv` et `solve`
//!   en taille n.
//! - **Cholesky** pour les matrices symétriques définies positives
//!   (matrices de covariance).
//! - **Type `Vector<N>`** : produit scalaire, produit vectoriel 3D, norme.
//! - **`#[must_use]`** sur les opérations arithmétiques.
//! - **Benchmarks en cycles** (compteur DWT) et CI de compilation embarquée.
//!
//! ## Détection des matrices singulières
//!
//! `inv` compare `|det|` à une borne **relative** à la taille des termes du
//! déterminant (et non à une constante absolue) : une matrice `1e-4 · I`
//! reste inversible, alors qu'une matrice de grands coefficients dont le
//! déterminant n'est que du bruit d'arrondi est correctement rejetée.

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;
mod matrix2x2;
mod matrix3x3;

pub use error::MatrixError;
pub use matrix2x2::Matrix2x2;
pub use matrix3x3::Matrix3x3;

/// Seuil relatif : `|det| <= SINGULAR_REL * borne` ⇒ matrice singulière.
/// `borne` est la somme des valeurs absolues des termes du déterminant.
const SINGULAR_REL: f32 = 8.0 * f32::EPSILON;

/// Valide un déterminant avant inversion.
///
/// `bound` : somme des valeurs absolues des termes qui composent `det`
/// (échelle de l'erreur d'arrondi attendue sur `det`).
#[inline]
pub(crate) fn check_det(det: f32, bound: f32) -> Result<(), MatrixError> {
    if !det.is_finite() || !bound.is_finite() {
        return Err(MatrixError::NonFiniteValue);
    }
    if det.abs() <= SINGULAR_REL * bound {
        Err(MatrixError::SingularMatrix)
    } else {
        Ok(())
    }
}