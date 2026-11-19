use core::fmt;

/// Error returned when a matrix operation is invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MatrixError {
    /// The matrix is singular (its determinant is zero or dominated by
    /// rounding error): inversion or solving (`inv`, `solve`) is impossible.
    SingularMatrix,
    /// The matrix (or the right-hand side of `solve`) contains NaN or ±∞,
    /// or the calculation overflowed.
    NonFiniteValue,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::SingularMatrix => f.write_str("matrice singulière : inversion ou résolution impossible"),
            MatrixError::NonFiniteValue => f.write_str("valeur non finie (NaN ou infini) ou débordement"),
        }
    }
}

impl core::error::Error for MatrixError {}