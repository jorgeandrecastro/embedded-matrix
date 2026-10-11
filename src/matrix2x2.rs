use crate::{check_det, MatrixError};
use core::ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign};

/// Matrice 2×2 en `f32`, stockée en row-major.
///
/// ```
/// use embedded_matrix::Matrix2x2;
///
/// let m = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
/// assert!((m.det() - (-2.0)).abs() < 1e-5);
/// assert_eq!(m[(1, 0)], 3.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Matrix2x2 {
    data: [[f32; 2]; 2],
}

impl Matrix2x2 {
    // ── Constructeurs ────────────────────────────────────────

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
        Self::new([[0.0; 2]; 2])
    }

    /// Matrice diagonale.
    #[inline]
    pub const fn from_diagonal(d: [f32; 2]) -> Self {
        Self::new([[d[0], 0.0], [0.0, d[1]]])
    }

    /// Matrice de rotation 2D d'angle `angle` (radians, sens trigonométrique).
    ///
    /// Disponible avec la feature `trig` (utilise `embedded-trig-f32`).
    /// Pour une rotation, l'inverse est la transposée : inutile d'appeler `inv`.
    ///
    /// ```
    /// use embedded_matrix::Matrix2x2;
    /// use embedded_trig_f32::consts::FRAC_PI_2;
    ///
    /// let r = Matrix2x2::rotation(FRAC_PI_2);
    /// let p = r * [1.0, 0.0];
    /// assert!(p[0].abs() < 1e-6 && (p[1] - 1.0).abs() < 1e-6);
    /// ```
    #[cfg(feature = "trig")]
    #[inline]
    pub fn rotation(angle: f32) -> Self {
        let (s, c) = embedded_trig_f32::sincos(angle);
        Self::new([[c, -s], [s, c]])
    }

    // ── Accès ────────────────────────────────────────────────

    /// Retourne l'élément à la ligne `row` et colonne `col` (0-indexé).
    ///
    /// # Panics
    /// Panique si `row` ou `col` vaut 2 ou plus.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row][col]
    }

    /// Donne accès aux données brutes row-major.
    #[inline]
    pub fn as_array(&self) -> &[[f32; 2]; 2] {
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
            core::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j])
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
    pub fn mul_vec(&self, v: [f32; 2]) -> [f32; 2] {
        let a = &self.data;
        [a[0][0] * v[0] + a[0][1] * v[1], a[1][0] * v[0] + a[1][1] * v[1]]
    }

    // ── Propriétés ───────────────────────────────────────────

    /// Transposée.
    #[inline]
    pub fn transpose(&self) -> Self {
        let a = &self.data;
        Self::new([[a[0][0], a[1][0]], [a[0][1], a[1][1]]])
    }

    /// Déterminant : `ad - bc`.
    #[inline]
    pub fn det(&self) -> f32 {
        let a = &self.data;
        a[0][0] * a[1][1] - a[0][1] * a[1][0]
    }

    /// Trace (somme des éléments diagonaux).
    #[inline]
    pub fn trace(&self) -> f32 {
        self.data[0][0] + self.data[1][1]
    }

    /// Résout le système `A·x = b` (formule de Cramer).
    ///
    /// Plus direct que `inv()` suivi d'un produit. Si vous avez plusieurs
    /// seconds membres pour la même matrice, calculez `inv()` une seule fois.
    ///
    /// ```
    /// use embedded_matrix::Matrix2x2;
    ///
    /// let a = Matrix2x2::new([[2.0, 1.0], [1.0, 3.0]]);
    /// let x = a.solve([3.0, 5.0]).unwrap();
    /// assert!((x[0] - 0.8).abs() < 1e-6 && (x[1] - 1.4).abs() < 1e-6);
    /// ```
    ///
    /// # Errors
    /// - [`MatrixError::SingularMatrix`] : même critère que [`inv`](Self::inv).
    /// - [`MatrixError::NonFiniteValue`] : NaN / ±∞ dans la matrice, dans `b`,
    ///   ou débordement.
    pub fn solve(&self, b: [f32; 2]) -> Result<[f32; 2], MatrixError> {
        let [[a, m], [c, d]] = self.data;
        let det = a * d - m * c;
        check_det(det, (a * d).abs() + (m * c).abs())?;
        let x = [(b[0] * d - m * b[1]) / det, (a * b[1] - c * b[0]) / det];
        if x[0].is_finite() && x[1].is_finite() {
            Ok(x)
        } else {
            Err(MatrixError::NonFiniteValue)
        }
    }

    /// Compare deux matrices terme à terme avec une tolérance absolue.
    /// Renvoie `false` si un terme est NaN.
    #[inline]
    pub fn approx_eq(&self, other: &Self, tol: f32) -> bool {
        let (a, b) = (&self.data, &other.data);
        (0..2).all(|i| (0..2).all(|j| (a[i][j] - b[i][j]).abs() <= tol))
    }

    /// Inverse de la matrice.
    ///
    /// # Errors
    /// - [`MatrixError::SingularMatrix`] si `|det|` est négligeable devant
    ///   `|ad| + |bc|` (seuil relatif, voir la doc de la crate).
    /// - [`MatrixError::NonFiniteValue`] si la matrice contient NaN / ±∞
    ///   ou si le calcul déborde.
    pub fn inv(&self) -> Result<Self, MatrixError> {
        let [[a, b], [c, d]] = self.data;
        let det = a * d - b * c;
        check_det(det, (a * d).abs() + (b * c).abs())?;
        let inv_det = 1.0 / det;
        Ok(Self::new([
            [d * inv_det, -b * inv_det],
            [-c * inv_det, a * inv_det],
        ]))
    }
}

// ── Conversions et accès par indice ──────────────────────────

impl From<[[f32; 2]; 2]> for Matrix2x2 {
    #[inline]
    fn from(data: [[f32; 2]; 2]) -> Self {
        Self::new(data)
    }
}

impl From<Matrix2x2> for [[f32; 2]; 2] {
    #[inline]
    fn from(m: Matrix2x2) -> Self {
        m.data
    }
}

/// `m[(ligne, colonne)]` : panique si hors limites.
impl Index<(usize, usize)> for Matrix2x2 {
    type Output = f32;
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &f32 {
        &self.data[row][col]
    }
}

// ── core::ops ────────────────────────────────────────────────

/// `a + b` : addition terme à terme.
impl Add for Matrix2x2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Matrix2x2::add(&self, &rhs)
    }
}

/// `a - b` : soustraction terme à terme.
impl Sub for Matrix2x2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Matrix2x2::sub(&self, &rhs)
    }
}

/// `a * b` : produit matriciel.
impl Mul for Matrix2x2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Matrix2x2::mul(&self, &rhs)
    }
}

/// `a * s` : multiplication par un scalaire `f32`.
impl Mul<f32> for Matrix2x2 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self {
        self.scale(s)
    }
}

/// `s * a` : multiplication par un scalaire `f32` à gauche.
impl Mul<Matrix2x2> for f32 {
    type Output = Matrix2x2;
    #[inline]
    fn mul(self, m: Matrix2x2) -> Matrix2x2 {
        m.scale(self)
    }
}

/// `a * v` : matrice × vecteur colonne.
impl Mul<[f32; 2]> for Matrix2x2 {
    type Output = [f32; 2];
    #[inline]
    fn mul(self, v: [f32; 2]) -> [f32; 2] {
        self.mul_vec(v)
    }
}

/// `-a` : négation terme à terme.
impl Neg for Matrix2x2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.scale(-1.0)
    }
}

impl AddAssign for Matrix2x2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Matrix2x2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for Matrix2x2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<f32> for Matrix2x2 {
    #[inline]
    fn mul_assign(&mut self, s: f32) {
        *self = self.scale(s);
    }
}