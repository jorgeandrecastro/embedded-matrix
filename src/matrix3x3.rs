use crate::{check_det, MatrixError};
use core::ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 3×3 `f32` matrix stored in row-major order.
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
    // ── Constructors ─────────────────────────────────────────

    /// Creates a matrix from a row-major array.
    #[inline]
    pub const fn new(data: [[f32; 3]; 3]) -> Self {
        Self { data }
    }

    /// Identity matrix.
    #[inline]
    pub const fn identity() -> Self {
        Self::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    /// Zero matrix.
    #[inline]
    pub const fn zero() -> Self {
        Self::new([[0.0; 3]; 3])
    }

    /// Diagonal matrix.
    #[inline]
    pub const fn from_diagonal(d: [f32; 3]) -> Self {
        Self::new([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]])
    }

    /// 3D rotation around the X axis (radians, right-hand rule).
    ///
    /// Available with the `trig` feature (uses `embedded-trig-f32`).
    #[cfg(feature = "trig")]
    #[inline]
    pub fn rotation_x(angle: f32) -> Self {
        let (s, c) = embedded_trig_f32::sincos(angle);
        Self::new([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    }

    /// 3D rotation around the Y axis (radians, right-hand rule).
    ///
    /// Available with the `trig` feature.
    #[cfg(feature = "trig")]
    #[inline]
    pub fn rotation_y(angle: f32) -> Self {
        let (s, c) = embedded_trig_f32::sincos(angle);
        Self::new([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    }

    /// 3D rotation around the Z axis (radians, right-hand rule).
    ///
    /// Available with the `trig` feature.
    ///
    /// ```
    /// use embedded_matrix::Matrix3x3;
    /// use embedded_trig_f32::consts::FRAC_PI_2;
    ///
    /// let r = Matrix3x3::rotation_z(FRAC_PI_2);
    /// let p = r * [1.0, 0.0, 5.0];
    /// assert!(p[0].abs() < 1e-6 && (p[1] - 1.0).abs() < 1e-6 && p[2] == 5.0);
    /// ```
    #[cfg(feature = "trig")]
    #[inline]
    pub fn rotation_z(angle: f32) -> Self {
        let (s, c) = embedded_trig_f32::sincos(angle);
        Self::new([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    }

    // ── Access ───────────────────────────────────────────────

    /// Returns the element at row `row` and column `col` (zero-indexed).
    ///
    /// # Panics
    /// Panics if `row` or `col` is 3 or greater.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row][col]
    }

    /// Provides access to the raw row-major data.
    #[inline]
    pub fn as_array(&self) -> &[[f32; 3]; 3] {
        &self.data
    }

    // ── Arithmetic operations ────────────────────────────────

    /// Element-wise addition.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn add(&self, rhs: &Self) -> Self {
        let (a, b) = (&self.data, &rhs.data);
        Self::new(core::array::from_fn(|i| core::array::from_fn(|j| a[i][j] + b[i][j])))
    }

    /// Element-wise subtraction.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn sub(&self, rhs: &Self) -> Self {
        let (a, b) = (&self.data, &rhs.data);
        Self::new(core::array::from_fn(|i| core::array::from_fn(|j| a[i][j] - b[i][j])))
    }

    /// Standard matrix multiplication.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(&self, rhs: &Self) -> Self {
        let (a, b) = (&self.data, &rhs.data);
        Self::new(core::array::from_fn(|i| {
            core::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
        }))
    }

    /// Scalar multiplication.
    #[inline]
    pub fn scale(&self, s: f32) -> Self {
        let a = &self.data;
        Self::new(core::array::from_fn(|i| core::array::from_fn(|j| a[i][j] * s)))
    }

    /// Matrix × column vector product.
    #[inline]
    pub fn mul_vec(&self, v: [f32; 3]) -> [f32; 3] {
        let a = &self.data;
        core::array::from_fn(|i| a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2])
    }

    // ── Properties ───────────────────────────────────────────

    /// Transpose.
    #[inline]
    pub fn transpose(&self) -> Self {
        let a = &self.data;
        Self::new(core::array::from_fn(|i| core::array::from_fn(|j| a[j][i])))
    }

    /// Determinant by expansion along the first row (cofactors).
    #[inline]
    pub fn det(&self) -> f32 {
        let d = &self.data;
        d[0][0] * (d[1][1] * d[2][2] - d[1][2] * d[2][1])
            - d[0][1] * (d[1][0] * d[2][2] - d[1][2] * d[2][0])
            + d[0][2] * (d[1][0] * d[2][1] - d[1][1] * d[2][0])
    }

    /// Trace (sum of the diagonal elements).
    #[inline]
    pub fn trace(&self) -> f32 {
        self.data[0][0] + self.data[1][1] + self.data[2][2]
    }

    /// Solves the system `A·x = b`.
    ///
    /// Computed as `A⁻¹·b`: same cost and singularity criterion as
    /// [`inv`](Self::inv). For multiple right-hand sides with the same
    /// matrix, compute `inv()` only once.
    ///
    /// ```
    /// use embedded_matrix::Matrix3x3;
    ///
    /// let a = Matrix3x3::new([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]);
    /// let x = a.solve([1.0, 2.0, 3.0]).unwrap();
    /// assert!((x[0] - 27.0).abs() < 1e-3);
    /// assert!((x[1] + 22.0).abs() < 1e-3);
    /// assert!((x[2] - 6.0).abs() < 1e-3);
    /// ```
    ///
    /// # Errors
    /// - [`MatrixError::SingularMatrix`]: same criterion as [`inv`](Self::inv).
    /// - [`MatrixError::NonFiniteValue`]: NaN / ±∞ in the matrix or `b`,
    ///   or overflow.
    pub fn solve(&self, b: [f32; 3]) -> Result<[f32; 3], MatrixError> {
        let x = self.inv()?.mul_vec(b);
        if x.iter().all(|v| v.is_finite()) {
            Ok(x)
        } else {
            Err(MatrixError::NonFiniteValue)
        }
    }

    /// Compares two matrices element-wise with an absolute tolerance.
    /// Returns `false` if an element is NaN.
    #[inline]
    pub fn approx_eq(&self, other: &Self, tol: f32) -> bool {
        let (a, b) = (&self.data, &other.data);
        (0..3).all(|i| (0..3).all(|j| (a[i][j] - b[i][j]).abs() <= tol))
    }

    /// Inverse using the cofactor method (adjugate matrix / determinant).
    ///
    /// # Errors
    /// - [`MatrixError::SingularMatrix`] if `|det|` is negligible compared to
    ///   the sum of the absolute values of the determinant terms (relative
    ///   threshold; see the crate documentation).
    /// - [`MatrixError::NonFiniteValue`] if the matrix contains NaN / ±∞
    ///   or the calculation overflows.
    pub fn inv(&self) -> Result<Self, MatrixError> {
        let d = &self.data;

        // First-row cofactors, reused for the determinant.
        let c00 = d[1][1] * d[2][2] - d[1][2] * d[2][1];
        let c01 = d[1][2] * d[2][0] - d[1][0] * d[2][2];
        let c02 = d[1][0] * d[2][1] - d[1][1] * d[2][0];

        let det = d[0][0] * c00 + d[0][1] * c01 + d[0][2] * c02;

        // Scale of the rounding error in `det`.
        let bound = d[0][0].abs() * ((d[1][1] * d[2][2]).abs() + (d[1][2] * d[2][1]).abs())
            + d[0][1].abs() * ((d[1][0] * d[2][2]).abs() + (d[1][2] * d[2][0]).abs())
            + d[0][2].abs() * ((d[1][0] * d[2][1]).abs() + (d[1][1] * d[2][0]).abs());

        check_det(det, bound)?;
        let inv_det = 1.0 / det;

        // Adjugate matrix (transpose of the cofactor matrix) / det.
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

// ── Conversions and index access ─────────────────────────────

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

/// `m[(row, column)]`: panics if out of bounds.
impl Index<(usize, usize)> for Matrix3x3 {
    type Output = f32;
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &f32 {
        &self.data[row][col]
    }
}

// ── core::ops ────────────────────────────────────────────────

/// `a + b`: element-wise addition.
impl Add for Matrix3x3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Matrix3x3::add(&self, &rhs)
    }
}

/// `a - b`: element-wise subtraction.
impl Sub for Matrix3x3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Matrix3x3::sub(&self, &rhs)
    }
}

/// `a * b`: matrix multiplication.
impl Mul for Matrix3x3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Matrix3x3::mul(&self, &rhs)
    }
}

/// `a * s`: multiplication by an `f32` scalar.
impl Mul<f32> for Matrix3x3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self {
        self.scale(s)
    }
}

/// `s * a`: multiplication by an `f32` scalar on the left.
impl Mul<Matrix3x3> for f32 {
    type Output = Matrix3x3;
    #[inline]
    fn mul(self, m: Matrix3x3) -> Matrix3x3 {
        m.scale(self)
    }
}

/// `a * v`: matrix × column vector.
impl Mul<[f32; 3]> for Matrix3x3 {
    type Output = [f32; 3];
    #[inline]
    fn mul(self, v: [f32; 3]) -> [f32; 3] {
        self.mul_vec(v)
    }
}

/// `-a`: element-wise negation.
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