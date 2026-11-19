// Copyright (C) 2026 Jorge Andre Castro
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the
// Free Software Foundation, either version 2 of the License, or
// any later version.

//! # embedded-matrix
//!
//! 2×2 and 3×3 `f32` matrices for `no_std` embedded systems.
//!
//! No default dependencies, no `unsafe`, and no allocation.
//!
//! ```rust
//! use embedded_matrix::{Matrix2x2, Matrix3x3, MatrixError};
//!
//! // Matrix multiplication with the * operator
//! let a = Matrix3x3::new([
//!     [1.0, 2.0, 3.0],
//!     [4.0, 5.0, 6.0],
//!     [7.0, 8.0, 9.0],
//! ]);
//! let b = Matrix3x3::identity();
//! assert_eq!(a * b, a);
//!
//! // Addition, subtraction, scalar multiplication
//! assert_eq!((a + b) - b, a);
//! let _double = a * 2.0;
//!
//! // Matrix × vector
//! let v = Matrix2x2::identity() * [1.0, 2.0];
//! assert_eq!(v, [1.0, 2.0]);
//!
//! // Inversion (Err if singular or non-finite)
//! match Matrix2x2::new([[2.0, 0.0], [0.0, 4.0]]).inv() {
//!     Ok(inv) => assert_eq!(inv, Matrix2x2::new([[0.5, 0.0], [0.0, 0.25]])),
//!     Err(MatrixError::SingularMatrix) => { /* handle error */ }
//!     Err(e) => panic!("{e}"),
//! }
//! ```
//!
//! ## Solving systems
//!
//! `solve(b)` solves `A·x = b` and returns `x` directly:
//!
//! ```rust
//! use embedded_matrix::Matrix2x2;
//!
//! let a = Matrix2x2::new([[2.0, 1.0], [1.0, 3.0]]);
//! let x = a.solve([3.0, 5.0]).unwrap(); // x = [0.8, 1.4]
//! assert!((x[0] - 0.8).abs() < 1e-6);
//! ```
//!
//! ## Optional feature: `trig`
//!
//! ```toml
//! embedded-matrix = { version = "0.3", features = ["trig"] }
//! ```
//!
//! Enables the `embedded-trig-f32` dependency and adds rotation constructors:
//! `Matrix2x2::rotation(angle)`, `Matrix3x3::rotation_x / _y / _z(angle)`
//! (angles in radians, right-hand rule). Without this feature, the crate
//! has **no dependencies**.
//!
//! ## Future ideas (not implemented)
//!
//! These ideas are not part of the current API; they are noted here
//! for the future:
//!
//! - **Arbitrary sizes**: stack-allocated `Matrix<const R: usize, const C: usize>`
//!   (alias `Matrix2x2 = Matrix<2, 2>`), with non-square matrix multiplication.
//! - **LU decomposition** with partial pivoting for `det`, `inv`, and `solve`
//!   at size n.
//! - **Cholesky** for symmetric positive-definite matrices
//!   (covariance matrices).
//! - **`Vector<N>` type**: dot product, 3D cross product, norm.
//! - **`#[must_use]`** on arithmetic operations.
//! - **Cycle benchmarks** (DWT counter) and CI builds for embedded targets.
//!
//! ## Singular matrix detection
//!
//! `inv` compares `|det|` against a bound **relative** to the magnitude of the
//! determinant terms (rather than an absolute constant): a `1e-4 · I` matrix
//! remains invertible, while a matrix with large coefficients whose
//! determinant is only rounding noise is correctly rejected.

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;
mod matrix2x2;
mod matrix3x3;

pub use error::MatrixError;
pub use matrix2x2::Matrix2x2;
pub use matrix3x3::Matrix3x3;

/// Relative threshold: `|det| <= SINGULAR_REL * bound` ⇒ singular matrix.
/// `bound` is the sum of the absolute values of the determinant terms.
const SINGULAR_REL: f32 = 8.0 * f32::EPSILON;

/// Validates a determinant before inversion.
///
/// `bound`: sum of the absolute values of the terms making up `det`
/// (the scale of the expected rounding error in `det`).
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