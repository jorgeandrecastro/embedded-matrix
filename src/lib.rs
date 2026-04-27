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
//! Sans dépendance, sans `unsafe`.
//!
//! ```rust
//! use embedded_matrix::{Matrix2x2, Matrix3x3, MatrixError};
//!
//! // Identité 2×2
//! let i2 = Matrix2x2::identity();
//!
//! // Produit matriciel avec l'opérateur *
//! let a = Matrix3x3::new([
//!     [1.0, 2.0, 3.0],
//!     [4.0, 5.0, 6.0],
//!     [7.0, 8.0, 9.0],
//! ]);
//! let b = Matrix3x3::identity();
//! let c = a * b;  // == a, syntaxe opérateur
//!
//! // Addition et soustraction
//! let d = a + b;
//! let e = d - b;
//!
//! // Scalaire
//! let f = a * 2.0;
//!
//! // Inversion (retourne Err si singulière)
//! match Matrix2x2::identity().inv() {
//!     Ok(inv) => { let _ = inv; }
//!     Err(MatrixError::SingularMatrix) => { /* gérer */ }
//! }
//! ```

#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
//  Erreur
/// Erreur retournée lors d'une opération invalide sur une matrice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixError {
    /// La matrice est singulière (déterminant nul) : inversion impossible.
    SingularMatrix,
}
//  Matrix2x2
/// Matrice 2×2 en `f32`, stockée en row-major.
///
/// ```
/// use embedded_matrix::Matrix2x2;
///
/// let m = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
/// assert!((m.det() - (-2.0)).abs() < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix2x2 {
    data: [[f32; 2]; 2],
}

impl Matrix2x2 {
    //  Constructeurs 

    /// Crée une matrice à partir d'un tableau row-major.
    #[inline]
    pub const fn new(data: [[f32; 2]; 2]) -> Self {
        Self { data }
    }

    /// Matrice identité.
    #[inline]
    pub const fn identity() -> Self {
        Self::new([[1.0, 0.0], [0.0, 1.0]])
    }

    /// Matrice nulle.
    #[inline]
    pub const fn zero() -> Self {
        Self::new([[0.0, 0.0], [0.0, 0.0]])
    }

    // Accès 
    /// Retourne l'élément à la ligne `row` et colonne `col` (0-indexé).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row][col]
    }

    /// Donne accès aux données brutes row-major.
    #[inline]
    pub fn as_array(&self) -> &[[f32; 2]; 2] {
        &self.data
    }

    // Opérations arithmétiques
    /// Addition terme à terme.
    #[inline]
    pub fn add(&self, rhs: &Self) -> Self {
        Self::new([
            [self.data[0][0] + rhs.data[0][0], self.data[0][1] + rhs.data[0][1]],
            [self.data[1][0] + rhs.data[1][0], self.data[1][1] + rhs.data[1][1]],
        ])
    }

    /// Soustraction terme à terme.
    #[inline]
    pub fn sub(&self, rhs: &Self) -> Self {
        Self::new([
            [self.data[0][0] - rhs.data[0][0], self.data[0][1] - rhs.data[0][1]],
            [self.data[1][0] - rhs.data[1][0], self.data[1][1] - rhs.data[1][1]],
        ])
    }

    /// Produit matriciel standard.
    #[inline]
    pub fn mul(&self, rhs: &Self) -> Self {
        let a = &self.data;
        let b = &rhs.data;
        Self::new([
            [
                a[0][0] * b[0][0] + a[0][1] * b[1][0],
                a[0][0] * b[0][1] + a[0][1] * b[1][1],
            ],
            [
                a[1][0] * b[0][0] + a[1][1] * b[1][0],
                a[1][0] * b[0][1] + a[1][1] * b[1][1],
            ],
        ])
    }

    /// Multiplication par un scalaire.
    #[inline]
    pub fn scale(&self, s: f32) -> Self {
        Self::new([
            [self.data[0][0] * s, self.data[0][1] * s],
            [self.data[1][0] * s, self.data[1][1] * s],
        ])
    }

    //  Propriétés 
    /// Transposée.
    #[inline]
    pub fn transpose(&self) -> Self {
        Self::new([
            [self.data[0][0], self.data[1][0]],
            [self.data[0][1], self.data[1][1]],
        ])
    }

    /// Déterminant : `ad - bc`.
    #[inline]
    pub fn det(&self) -> f32 {
        self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
    }

    /// Trace (somme des éléments diagonaux).
    #[inline]
    pub fn trace(&self) -> f32 {
        self.data[0][0] + self.data[1][1]
    }

    /// Inverse de la matrice.
    ///
    /// Retourne `Err(MatrixError::SingularMatrix)` si le déterminant est nul
    /// (seuil : `|det| < 1e-7`).
    pub fn inv(&self) -> Result<Self, MatrixError> {
        let d = self.det();
        if (if d < 0.0 { -d } else { d }) < 1e-7 {
            return Err(MatrixError::SingularMatrix);
        }
        let inv_d = 1.0 / d;
        Ok(Self::new([
            [ self.data[1][1] * inv_d, -self.data[0][1] * inv_d],
            [-self.data[1][0] * inv_d,  self.data[0][0] * inv_d],
        ]))
    }
}

// core::ops pour Matrix2x2 
/// `a + b`  →  addition terme à terme.
impl core::ops::Add for Matrix2x2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Matrix2x2::add(&self, &rhs)
    }
}

/// `a - b`  →  soustraction terme à terme.
impl core::ops::Sub for Matrix2x2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Matrix2x2::sub(&self, &rhs)
    }
}

/// `a * b`  →  produit matriciel.
impl core::ops::Mul for Matrix2x2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Matrix2x2::mul(&self, &rhs)
    }
}

/// `a * s`  →  multiplication par un scalaire `f32`.
impl core::ops::Mul<f32> for Matrix2x2 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self {
        self.scale(s)
    }
}

/// `-a`  →  négation terme à terme.
impl core::ops::Neg for Matrix2x2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.scale(-1.0)
    }
}


//  Matrix3x3
/// Matrice 3×3 en `f32`, stockée en row-major.
///
/// ```
/// use embedded_matrix::Matrix3x3;
///
/// let m = Matrix3x3::identity();
/// assert!((m.det() - 1.0).abs() < 1e-5);
/// assert!((m.trace() - 3.0).abs() < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix3x3 {
    data: [[f32; 3]; 3],
}

impl Matrix3x3 {
    // Constructeurs 

    /// Crée une matrice à partir d'un tableau row-major.
    #[inline]
    pub const fn new(data: [[f32; 3]; 3]) -> Self {
        Self { data }
    }

    /// Matrice identité.
    #[inline]
    pub const fn identity() -> Self {
        Self::new([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    }

    /// Matrice nulle.
    #[inline]
    pub const fn zero() -> Self {
        Self::new([[0.0; 3]; 3])
    }

    //  Accès 
    /// Retourne l'élément à la ligne `row` et colonne `col` (0-indexé).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row][col]
    }

    /// Donne accès aux données brutes row-major.
    #[inline]
    pub fn as_array(&self) -> &[[f32; 3]; 3] {
        &self.data
    }

    //  Opérations arithmétiques 
    /// Additionne cette matrice avec une autre matrice terme à terme.
    ///
    /// # Arguments
    ///
    /// * `rhs`  La matrice à additionner (membre droit)
    ///
    /// # Retours
    ///
    /// Une nouvelle matrice contenant la somme terme à terme
    #[inline]
    pub fn add(&self, rhs: &Self) -> Self {
        let mut out = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }
        Self::new(out)
    }

    /// Soustraction terme à terme.
    #[inline]
    pub fn sub(&self, rhs: &Self) -> Self {
        let mut out = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.data[i][j] - rhs.data[i][j];
            }
        }
        Self::new(out)
    }

    /// Produit matriciel standard.
    #[inline]
    pub fn mul(&self, rhs: &Self) -> Self {
        let mut out = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    out[i][j] += self.data[i][k] * rhs.data[k][j];
                }
            }
        }
        Self::new(out)
    }

    /// Multiplication par un scalaire.
    #[inline]
    pub fn scale(&self, s: f32) -> Self {
        let mut out = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.data[i][j] * s;
            }
        }
        Self::new(out)
    }

    //  Propriétés 
    /// Transposée.
    #[inline]
    pub fn transpose(&self) -> Self {
        let mut out = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[j][i] = self.data[i][j];
            }
        }
        Self::new(out)
    }

    /// Déterminant par la règle de Sarrus.
    #[inline]
    pub fn det(&self) -> f32 {
        let d = &self.data;
        d[0][0] * (d[1][1] * d[2][2] - d[1][2] * d[2][1])
            - d[0][1] * (d[1][0] * d[2][2] - d[1][2] * d[2][0])
            + d[0][2] * (d[1][0] * d[2][1] - d[1][1] * d[2][0])
    }

    /// Trace (somme des éléments diagonaux).
    #[inline]
    pub fn trace(&self) -> f32 {
        self.data[0][0] + self.data[1][1] + self.data[2][2]
    }

    /// Inverse par la méthode des cofacteurs.
    ///
    /// Retourne `Err(MatrixError::SingularMatrix)` si `|det| < 1e-7`.
    pub fn inv(&self) -> Result<Self, MatrixError> {
        let det = self.det();
        if (if det < 0.0 { -det } else { det }) < 1e-7 {
            return Err(MatrixError::SingularMatrix);
        }
        let d = &self.data;
        let inv_det = 1.0 / det;

        // Matrice des cofacteurs transposée (= matrice adjointe)
        Ok(Self::new([
            [
                (d[1][1] * d[2][2] - d[1][2] * d[2][1]) * inv_det,
                (d[0][2] * d[2][1] - d[0][1] * d[2][2]) * inv_det,
                (d[0][1] * d[1][2] - d[0][2] * d[1][1]) * inv_det,
            ],
            [
                (d[1][2] * d[2][0] - d[1][0] * d[2][2]) * inv_det,
                (d[0][0] * d[2][2] - d[0][2] * d[2][0]) * inv_det,
                (d[0][2] * d[1][0] - d[0][0] * d[1][2]) * inv_det,
            ],
            [
                (d[1][0] * d[2][1] - d[1][1] * d[2][0]) * inv_det,
                (d[0][1] * d[2][0] - d[0][0] * d[2][1]) * inv_det,
                (d[0][0] * d[1][1] - d[0][1] * d[1][0]) * inv_det,
            ],
        ]))
    }
}

//  core::ops pour Matrix3x3 

/// `a + b`  →  addition terme à terme.
impl core::ops::Add for Matrix3x3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Matrix3x3::add(&self, &rhs)
    }
}

/// `a - b`  →  soustraction terme à terme.
impl core::ops::Sub for Matrix3x3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Matrix3x3::sub(&self, &rhs)
    }
}

/// `a * b`  →  produit matriciel.
impl core::ops::Mul for Matrix3x3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Matrix3x3::mul(&self, &rhs)
    }
}

/// `a * s`  →  multiplication par un scalaire `f32`.
impl core::ops::Mul<f32> for Matrix3x3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self {
        self.scale(s)
    }
}

/// `-a`  →  négation terme à terme.
impl core::ops::Neg for Matrix3x3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.scale(-1.0)
    }
}


//  Tests

#[cfg(test)]
mod tests {
    use super::*;

    //  Matrix2x2 
    #[test]
    fn m2_identity_det() {
        assert!((Matrix2x2::identity().det() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn m2_det() {
        let m = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
        assert!((m.det() - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn m2_trace() {
        let m = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
        assert!((m.trace() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn m2_add_sub() {
        let a = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix2x2::new([[5.0, 6.0], [7.0, 8.0]]);
        let s = a.add(&b);
        assert!((s.get(0, 0) - 6.0).abs() < 1e-6);
        assert!((s.get(1, 1) - 12.0).abs() < 1e-6);
        let d = b.sub(&a);
        assert!((d.get(0, 0) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn m2_mul_identity() {
        let m = Matrix2x2::new([[3.0, 1.0], [2.0, 4.0]]);
        let r = m.mul(&Matrix2x2::identity());
        assert!((r.get(0, 0) - m.get(0, 0)).abs() < 1e-6);
        assert!((r.get(1, 1) - m.get(1, 1)).abs() < 1e-6);
    }

    #[test]
    fn m2_mul() {
        let a = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix2x2::new([[2.0, 0.0], [1.0, 3.0]]);
        let c = a.mul(&b);
        // [1*2+2*1, 1*0+2*3] = [4, 6]
        // [3*2+4*1, 3*0+4*3] = [10, 12]
        assert!((c.get(0, 0) - 4.0).abs() < 1e-6);
        assert!((c.get(0, 1) - 6.0).abs() < 1e-6);
        assert!((c.get(1, 0) - 10.0).abs() < 1e-6);
        assert!((c.get(1, 1) - 12.0).abs() < 1e-6);
    }

    #[test]
    fn m2_scale() {
        let m = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
        let s = m.scale(2.0);
        assert!((s.get(0, 0) - 2.0).abs() < 1e-6);
        assert!((s.get(1, 1) - 8.0).abs() < 1e-6);
    }

    #[test]
    fn m2_transpose() {
        let m = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
        let t = m.transpose();
        assert!((t.get(0, 1) - 3.0).abs() < 1e-6);
        assert!((t.get(1, 0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn m2_inv_ok() {
        let m = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
        let inv = m.inv().unwrap();
        // M * M^-1 doit être proche de l'identité
        let i = m.mul(&inv);
        assert!((i.get(0, 0) - 1.0).abs() < 1e-5);
        assert!((i.get(1, 1) - 1.0).abs() < 1e-5);
        assert!(i.get(0, 1).abs() < 1e-5);
        assert!(i.get(1, 0).abs() < 1e-5);
    }

    #[test]
    fn m2_inv_singular() {
        let m = Matrix2x2::new([[1.0, 2.0], [2.0, 4.0]]); // det = 0
        assert_eq!(m.inv(), Err(MatrixError::SingularMatrix));
    }

    //  Matrix3x3 

    #[test]
    fn m3_identity_det() {
        assert!((Matrix3x3::identity().det() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn m3_identity_trace() {
        assert!((Matrix3x3::identity().trace() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn m3_det() {
        let m = Matrix3x3::new([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);
        // Matrice singulière, det = 0
        assert!(m.det().abs() < 1e-4);
    }

    #[test]
    fn m3_det_nonzero() {
        let m = Matrix3x3::new([
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 2.0],
        ]);
        assert!((m.det() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn m3_add_sub() {
        let a = Matrix3x3::identity();
        let b = Matrix3x3::identity();
        let s = a.add(&b);
        assert!((s.get(0, 0) - 2.0).abs() < 1e-6);
        let d = s.sub(&a);
        assert!((d.get(0, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn m3_mul_identity() {
        let m = Matrix3x3::new([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [5.0, 6.0, 0.0],
        ]);
        let r = m.mul(&Matrix3x3::identity());
        for i in 0..3 {
            for j in 0..3 {
                assert!((r.get(i, j) - m.get(i, j)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn m3_transpose() {
        let m = Matrix3x3::new([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);
        let t = m.transpose();
        assert!((t.get(0, 1) - 4.0).abs() < 1e-6);
        assert!((t.get(1, 0) - 2.0).abs() < 1e-6);
        assert!((t.get(2, 0) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn m3_inv_ok() {
        let m = Matrix3x3::new([
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 2.0],
        ]);
        let inv = m.inv().unwrap();
        let i = m.mul(&inv);
        for r in 0..3 {
            for c in 0..3 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!((i.get(r, c) - expected).abs() < 1e-5,
                    "i[{r}][{c}] = {} expected {expected}", i.get(r, c));
            }
        }
    }

    #[test]
    fn m3_inv_singular() {
        let m = Matrix3x3::new([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);
        assert_eq!(m.inv(), Err(MatrixError::SingularMatrix));
    }

    #[test]
    fn m3_scale() {
        let m = Matrix3x3::identity();
        let s = m.scale(3.0);
        assert!((s.get(0, 0) - 3.0).abs() < 1e-6);
        assert!((s.get(0, 1)).abs() < 1e-6);
    }

    // Opérateurs Matrix2x2
    #[test]
    fn m2_ops_add() {
        let a = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix2x2::new([[1.0, 0.0], [0.0, 1.0]]);
        let c = a + b;
        assert!((c.get(0, 0) - 2.0).abs() < 1e-6);
        assert!((c.get(1, 1) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn m2_ops_sub() {
        let a = Matrix2x2::new([[3.0, 2.0], [1.0, 4.0]]);
        let b = Matrix2x2::new([[1.0, 1.0], [1.0, 1.0]]);
        let c = a - b;
        assert!((c.get(0, 0) - 2.0).abs() < 1e-6);
        assert!((c.get(1, 1) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn m2_ops_mul_matrix() {
        let a = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix2x2::new([[2.0, 0.0], [1.0, 3.0]]);
        let c = a * b;
        assert!((c.get(0, 0) - 4.0).abs() < 1e-6);
        assert!((c.get(0, 1) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn m2_ops_mul_scalar() {
        let a = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = a * 3.0;
        assert!((b.get(0, 0) - 3.0).abs() < 1e-6);
        assert!((b.get(1, 1) - 12.0).abs() < 1e-6);
    }

    #[test]
    fn m2_ops_neg() {
        let a = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = -a;
        assert!((b.get(0, 0) + 1.0).abs() < 1e-6);
        assert!((b.get(1, 1) + 4.0).abs() < 1e-6);
    }

    //  Opérateurs Matrix3x3 

    #[test]
    fn m3_ops_add() {
        let a = Matrix3x3::identity();
        let b = Matrix3x3::identity();
        let c = a + b;
        assert!((c.get(0, 0) - 2.0).abs() < 1e-6);
        assert!((c.get(0, 1)).abs() < 1e-6);
    }

    #[test]
    fn m3_ops_sub() {
        let a = Matrix3x3::identity();
        let b = Matrix3x3::identity();
        let c = a - b;
        assert!(c.get(0, 0).abs() < 1e-6);
    }

    #[test]
    fn m3_ops_mul_matrix() {
        let a = Matrix3x3::new([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [5.0, 6.0, 0.0],
        ]);
        let i = Matrix3x3::identity();
        let c = a * i;
        for r in 0..3 {
            for col in 0..3 {
                assert!((c.get(r, col) - a.get(r, col)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn m3_ops_mul_scalar() {
        let a = Matrix3x3::identity();
        let b = a * 5.0;
        assert!((b.get(0, 0) - 5.0).abs() < 1e-6);
        assert!((b.get(0, 1)).abs() < 1e-6);
    }

    #[test]
    fn m3_ops_neg() {
        let a = Matrix3x3::identity();
        let b = -a;
        assert!((b.get(0, 0) + 1.0).abs() < 1e-6);
        assert!((b.get(1, 1) + 1.0).abs() < 1e-6);
    }
}