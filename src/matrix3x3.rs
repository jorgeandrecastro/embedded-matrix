use crate::{check_det, MatrixError};
use core::ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign};

/// Matrice 3×3 en `f32`, stockée en row-major.
///
/// ```
/// use embedded_matrix::Matrix3x3;
///
/// let m = Matrix3x3::identity();
/// assert!((m.det() - 1.0).abs() < 1e-5);
/// assert!((m.trace() - 3.0).abs() < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Matrix3x3 {
    data: [[f32; 3]; 3],
}

impl Matrix3x3 {
    // ── Constructeurs ────────────────────────────────────────

    /// Crée une matrice à partir d'un tableau row-major.
    #[inline]
    pub const fn new(data: [[f32; 3]; 3]) -> Self {
        Self { data }
    }

    /// Matrice identité.
    #[inline]
    pub const fn identity() -> Self {
        Self::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    /// Matrice nulle.
    #[inline]
    pub const fn zero() -> Self {
        Self::new([[0.0; 3]; 3])
    }

    /// Matrice diagonale.
    #[inline]
    pub const fn from_diagonal(d: [f32; 3]) -> Self {
        Self::new([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]])
    }

    // ── Accès ────────────────────────────────────────────────

    /// Retourne l'élément à la ligne `row` et colonne `col` (0-indexé).
    ///
    /// # Panics
    /// Panique si `row` ou `col` vaut 3 ou plus.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row][col]
    }

    /// Donne accès aux données brutes row-major.
    #[inline]
    pub fn as_array(&self) -> &[[f32; 3]; 3] {
        &self.data
    }

    // ── Opérations arithmétiques ─────────────────────────────

    /// Addition terme à terme.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn add(&self, rhs: &Self) -> Self {
        let (a, b) = (&self.data, &rhs.data);
        Self::new(core::array::from_fn(|i| core::array::from_fn(|j| a[i][j] + b[i][j])))
    }

    /// Soustraction terme à terme.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn sub(&self, rhs: &Self) -> Self {
        let (a, b) = (&self.data, &rhs.data);
        Self::new(core::array::from_fn(|i| core::array::from_fn(|j| a[i][j] - b[i][j])))
    }

    /// Produit matriciel standard.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(&self, rhs: &Self) -> Self {
        let (a, b) = (&self.data, &rhs.data);
        Self::new(core::array::from_fn(|i| {
            core::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
        }))
    }

    /// Multiplication par un scalaire.
    #[inline]
    pub fn scale(&self, s: f32) -> Self {
        let a = &self.data;
        Self::new(core::array::from_fn(|i| core::array::from_fn(|j| a[i][j] * s)))
    }

    /// Produit matrice × vecteur colonne.
    #[inline]
    pub fn mul_vec(&self, v: [f32; 3]) -> [f32; 3] {
        let a = &self.data;
        core::array::from_fn(|i| a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2])
    }

    // ── Propriétés ───────────────────────────────────────────

    /// Transposée.
    #[inline]
    pub fn transpose(&self) -> Self {
        let a = &self.data;
        Self::new(core::array::from_fn(|i| core::array::from_fn(|j| a[j][i])))
    }

    /// Déterminant par développement selon la première ligne (cofacteurs).
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

    /// Compare deux matrices terme à terme avec une tolérance absolue.
    /// Renvoie `false` si un terme est NaN.
    #[inline]
    pub fn approx_eq(&self, other: &Self, tol: f32) -> bool {
        let (a, b) = (&self.data, &other.data);
        (0..3).all(|i| (0..3).all(|j| (a[i][j] - b[i][j]).abs() <= tol))
    }

    /// Inverse par la méthode des cofacteurs (matrice adjointe / déterminant).
    ///
    /// # Errors
    /// - [`MatrixError::SingularMatrix`] si `|det|` est négligeable devant la
    ///   somme des valeurs absolues des termes du déterminant (seuil relatif,
    ///   voir la doc de la crate).
    /// - [`MatrixError::NonFiniteValue`] si la matrice contient NaN / ±∞
    ///   ou si le calcul déborde.
    pub fn inv(&self) -> Result<Self, MatrixError> {
        let d = &self.data;

        // Cofacteurs de la première ligne : réutilisés pour le déterminant.
        let c00 = d[1][1] * d[2][2] - d[1][2] * d[2][1];
        let c01 = d[1][2] * d[2][0] - d[1][0] * d[2][2];
        let c02 = d[1][0] * d[2][1] - d[1][1] * d[2][0];

        let det = d[0][0] * c00 + d[0][1] * c01 + d[0][2] * c02;

        // Échelle de l'erreur d'arrondi sur `det`.
        let bound = d[0][0].abs() * ((d[1][1] * d[2][2]).abs() + (d[1][2] * d[2][1]).abs())
            + d[0][1].abs() * ((d[1][0] * d[2][2]).abs() + (d[1][2] * d[2][0]).abs())
            + d[0][2].abs() * ((d[1][0] * d[2][1]).abs() + (d[1][1] * d[2][0]).abs());

        check_det(det, bound)?;
        let inv_det = 1.0 / det;

        // Matrice adjointe (transposée de la matrice des cofacteurs) / det.
        Ok(Self::new([
            [
                c00 * inv_det,
                (d[0][2] * d[2][1] - d[0][1] * d[2][2]) * inv_det,
                (d[0][1] * d[1][2] - d[0][2] * d[1][1]) * inv_det,
            ],
            [
                c01 * inv_det,
                (d[0][0] * d[2][2] - d[0][2] * d[2][0]) * inv_det,
                (d[0][2] * d[1][0] - d[0][0] * d[1][2]) * inv_det,
            ],
            [
                c02 * inv_det,
                (d[0][1] * d[2][0] - d[0][0] * d[2][1]) * inv_det,
                (d[0][0] * d[1][1] - d[0][1] * d[1][0]) * inv_det,
            ],
        ]))
    }
}

// ── Conversions et accès par indice ──────────────────────────

impl From<[[f32; 3]; 3]> for Matrix3x3 {
    #[inline]
    fn from(data: [[f32; 3]; 3]) -> Self {
        Self::new(data)
    }
}

impl From<Matrix3x3> for [[f32; 3]; 3] {
    #[inline]
    fn from(m: Matrix3x3) -> Self {
        m.data
    }
}

/// `m[(ligne, colonne)]` : panique si hors limites.
impl Index<(usize, usize)> for Matrix3x3 {
    type Output = f32;
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &f32 {
        &self.data[row][col]
    }
}

// ── core::ops ────────────────────────────────────────────────

/// `a + b` : addition terme à terme.
impl Add for Matrix3x3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Matrix3x3::add(&self, &rhs)
    }
}

/// `a - b` : soustraction terme à terme.
impl Sub for Matrix3x3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Matrix3x3::sub(&self, &rhs)
    }
}

/// `a * b` : produit matriciel.
impl Mul for Matrix3x3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Matrix3x3::mul(&self, &rhs)
    }
}

/// `a * s` : multiplication par un scalaire `f32`.
impl Mul<f32> for Matrix3x3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self {
        self.scale(s)
    }
}

/// `s * a` : multiplication par un scalaire `f32` à gauche.
impl Mul<Matrix3x3> for f32 {
    type Output = Matrix3x3;
    #[inline]
    fn mul(self, m: Matrix3x3) -> Matrix3x3 {
        m.scale(self)
    }
}

/// `a * v` : matrice × vecteur colonne.
impl Mul<[f32; 3]> for Matrix3x3 {
    type Output = [f32; 3];
    #[inline]
    fn mul(self, v: [f32; 3]) -> [f32; 3] {
        self.mul_vec(v)
    }
}

/// `-a` : négation terme à terme.
impl Neg for Matrix3x3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.scale(-1.0)
    }
}

impl AddAssign for Matrix3x3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Matrix3x3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for Matrix3x3 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<f32> for Matrix3x3 {
    #[inline]
    fn mul_assign(&mut self, s: f32) {
        *self = self.scale(s);
    }
}