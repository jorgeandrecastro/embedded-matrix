use core::fmt;

/// Erreur retournée lors d'une opération invalide sur une matrice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MatrixError {
    /// La matrice est singulière (déterminant nul ou noyé dans l'erreur
    /// d'arrondi) : inversion ou résolution (`inv`, `solve`) impossible.
    SingularMatrix,
    /// La matrice (ou le second membre de `solve`) contient NaN ou ±∞,
    /// ou le calcul a débordé.
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