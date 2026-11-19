# Changelog

## [0.3.0] - 2026-10-11

### Added
- `solve(b)`: solve `A·x = b` for 2×2 and 3×3 matrices, with the same singularity criterion as `inv`.
- Optional Cargo feature `trig` (dependency on `embedded-trig-f32`): `Matrix2x2::rotation`, `Matrix3x3::rotation_x`, `rotation_y`, `rotation_z`. No dependencies without the feature.
- Tests for `solve` and, with the `trig` feature, rotations (determinant 1, orthogonality, composition, right-hand rule).
- Documentation: "Future ideas" section (size n, LU, Cholesky, `Vector<N>`, `#[must_use]`, benchmarks), not implemented.

## [0.2.0] - 2026-10-11

### Fixed
- **Relative singularity threshold**: `inv` no longer uses `|det| < 1e-7` (which depends on scale). A matrix is singular if `|det| <= 8·EPSILON × (sum of determinant |terms|)`. `1e-4 · I` is now invertible; matrices with large coefficients whose determinant is only rounding noise are rejected; the zero matrix is singular.
- `inv` no longer returns a matrix of NaNs when the input contains NaN or ±∞.
- `Matrix3x3::det` documentation: cofactor expansion (not "Sarrus' rule").

### Changed (breaking)
- `MatrixError` is now `#[non_exhaustive]` and has a new `NonFiniteValue` variant: add an `Err(_)` arm to your `match` expressions.
- Minimum Rust version: 1.85 (2024 edition).

### Added
- `MatrixError` implements `Display` and `core::error::Error`.
- Matrix × vector product: `mul_vec([f32; N])` and `matrix * [f32; N]`.
- `scalar * matrix`, `+=`, `-=`, `*=` (matrix and scalar).
- Index access `m[(row, column)]`.
- `From<[[f32; N]; N]>` and the reverse conversion, `Default` (zero matrix), `from_diagonal`, `approx_eq`.

### Improved
- `Matrix3x3::inv` reuses the first-row cofactors to calculate the determinant.
- Code split into modules (`error`, `matrix2x2`, `matrix3x3`); tests moved to `tests/matrix.rs` (known values, robustness, properties of pseudorandom matrices).
- The `add`, `sub`, and `mul` methods remain available for compatibility.

## [0.1.0]

- Initial release: `Matrix2x2` and `Matrix3x3` (`new`, `identity`, `zero`, `get`, `as_array`, `add`, `sub`, `mul`, `scale`, `transpose`, `det`, `trace`, `inv`), operators `+`, `-`, `*`, `-a`.