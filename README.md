# embedded-matrix

2×2 and 3×3 `f32` matrices for `no_std` embedded systems.

[![Crates.io](https://img.shields.io/crates/v/embedded-matrix.svg)](https://crates.io/crates/embedded-matrix)
[![License](https://img.shields.io/crates/l/embedded-matrix.svg)](LICENSE)
[![Docs.rs](https://docs.rs/embedded-matrix/badge.svg)](https://docs.rs/embedded-matrix/)

Version 0.3 adds `solve` (solving `A·x = b`) and, behind the optional `trig` feature, rotation constructors. Version 0.2 fixed the singular matrix detection threshold (now **relative to scale**), added the `NonFiniteValue` error, matrix × vector multiplication, and several operators. See the [CHANGELOG](CHANGELOG.md).

## Highlights

- **Zero default dependencies**: no libm, no micromath, no C library, no allocation.
- **Safety**: `#![forbid(unsafe_code)]` and inversion via `Result<_, MatrixError>`.
- **Portability**: works on Cortex-M0+, M4F, M33, RISC-V, and any `no_std` target.
- **Native operators**: `a * b`, `a + b`, `a - b`, `a * 2.0`, `2.0 * a`, `-a`, `a * [x, y, z]`, `+=`, `-=`, `*=`.
- **Robust inversion**: relative singularity threshold, explicit rejection of NaN and ±∞.

## Usage

```toml
[dependencies]
embedded-matrix = "0.4"
```

Minimum Rust version: **1.85** (2024 edition).

### Optional `trig` feature

```toml
[dependencies]
embedded-matrix = { version = "0.4", features = ["trig"] }
```

Enables the [`embedded-trig-f32`](https://crates.io/crates/embedded-trig-f32) dependency and rotation constructors (angles in radians, right-hand rule). **Without this feature, the crate has no dependencies.**

```rust
use embedded_matrix::{Matrix2x2, Matrix3x3};

let r = Matrix2x2::rotation(0.5);
let p = r * [1.0, 0.0];

let rz = Matrix3x3::rotation_z(0.5);
let q = rz * [1.0, 0.0, 5.0];   // z remains unchanged
```

For a rotation, the inverse is the transpose: `r.transpose()` is sufficient; there is no need to call `inv()`.

## Available operations

| Method               | Operator        | Description                                             |
|----------------------|-----------------|---------------------------------------------------------|
| `new(data)`          | `From<[[f32; N]; N]>` | Construct from a row-major array                  |
| `identity()`         | —               | Identity matrix                                         |
| `zero()`             | `Default`       | Zero matrix                                             |
| `from_diagonal(d)`   | —               | Diagonal matrix                                         |
| `add(&m)`            | `a + b`, `+=`   | Element-wise addition                                   |
| `sub(&m)`            | `a - b`, `-=`   | Element-wise subtraction                                |
| `mul(&m)`            | `a * b`, `*=`   | Matrix multiplication                                   |
| `scale(f32)`         | `a * 2.0`, `2.0 * a`, `*= 2.0` | Scalar multiplication                    |
| `mul_vec([f32; N])`  | `a * v`         | Matrix × column vector product                          |
| `solve([f32; N])`    | —               | Solve `A·x = b` → `Result<[f32; N], MatrixError>`      |
| `rotation(angle)` ¹  | —               | 2D rotation (`Matrix2x2`)                               |
| `rotation_x/y/z(angle)` ¹ | —          | 3D rotations (`Matrix3x3`)                              |
| —                    | `-a`            | Element-wise negation                                   |
| `transpose()`        | —               | Transpose                                               |
| `det()`              | —               | Determinant                                             |
| `trace()`            | —               | Trace (sum of the diagonal)                             |
| `inv()`              | —               | Inverse → `Result<M, MatrixError>`                       |
| `get(row, col)`      | `m[(row, col)]` | Element access (panics if out of bounds)                |
| `as_array()`         | `Into<[[f32; N]; N]>` | Access to row-major data                          |
| `approx_eq(&m, tol)` | —               | Equality within an absolute tolerance                   |

¹ Available with the `trig` feature (see above).

### Errors

| Variant                        | Case                                                             |
|--------------------------------|------------------------------------------------------------------|
| `MatrixError::SingularMatrix`  | Determinant is zero or dominated by rounding error               |
| `MatrixError::NonFiniteValue`  | The matrix contains NaN or ±∞, or the calculation overflowed    |

`MatrixError` is `#[non_exhaustive]`: include an `Err(_)` arm in your `match` expressions. It implements `Display` and `core::error::Error`.

## Examples

### Operator syntax

```rust
use embedded_matrix::Matrix2x2;

let a = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
let b = Matrix2x2::identity();

let c = a * b;      // matrix multiplication
let d = a + b;      // element-wise addition
let e = a - b;      // element-wise subtraction
let f = a * 2.0;    // scalar
let g = 2.0 * a;    // scalar on the left
let h = -a;         // negation
let v = a * [1.0, 1.0]; // matrix × vector → [3.0, 7.0]

let mut m = a;
m += b;
m *= 0.5;
```

### Complete example

```rust
use embedded_matrix::{Matrix2x2, Matrix3x3, MatrixError};

// 2×2 matrix
let a = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
let det = a.det();   // -2.0

match a.inv() {
    Ok(inv) => { let _ = a * inv; /* ≈ identity */ }
    Err(MatrixError::SingularMatrix) => { /* singular matrix */ }
    Err(_) => { /* NaN / infinity */ }
}

// 3×3 matrix
let r = Matrix3x3::new([
    [ 2.0, -1.0,  0.0],
    [-1.0,  2.0, -1.0],
    [ 0.0, -1.0,  2.0],
]);
let inv = r.inv().unwrap();
let trace = r.trace();  // 6.0
let det   = r.det();    // 4.0
let x = r * [1.0, 0.0, -1.0]; // matrix × vector
```

### Solving a system `A·x = b`

```rust
use embedded_matrix::{Matrix3x3, MatrixError};

let a = Matrix3x3::new([
    [1.0, 2.0, 3.0],
    [0.0, 1.0, 4.0],
    [5.0, 6.0, 0.0],
]);

match a.solve([1.0, 2.0, 3.0]) {
    Ok(x) => { /* x ≈ [27.0, -22.0, 6.0] */ let _ = x; }
    Err(MatrixError::SingularMatrix) => { /* no unique solution */ }
    Err(_) => { /* NaN / infinity */ }
}
```

`solve` applies the same singularity criterion as `inv()`. For 2×2 matrices it directly uses Cramer's rule; for 3×3 matrices it computes `A⁻¹·b` (at the same cost as `inv`). For multiple right-hand sides with the same matrix, call `inv()` only once.

### Embedded example: Frobenius norm with embedded-f32-sqrt (RP2350)

```rust
use embedded_matrix::Matrix3x3;
use embedded_f32_sqrt::sqrt;

// Frobenius norm: sqrt(Σ aij²)
fn frobenius_norm(m: &Matrix3x3) -> f32 {
    let mut sum = 0.0f32;
    for row in m.as_array() {
        for &v in row {
            sum += v * v;
        }
    }
    sqrt(sum).unwrap_or(0.0)
}
```

## Algorithms

- **`Matrix2x2`**: inversion using the direct formula, `1/det * [[d, -b], [-c, a]]`.
- **`Matrix3x3`**: determinant by expansion along the first row (cofactors); inversion using the cofactor method (adjugate matrix divided by the determinant), reusing the first-row cofactors for the determinant.

### Singular matrix detection

The threshold is **relative**: a matrix is considered singular if

```
|det| <= 8 · f32::EPSILON · B
```

where `B` is the sum of the absolute values of the determinant terms (`|ad| + |bc|` for 2×2, the six products of the expansion for 3×3). `B` gives the scale of the expected rounding error in `det`.

Consequences:

- `1e-4 · I` remains invertible (the old absolute threshold `|det| < 1e-7` incorrectly rejected it).
- A matrix with large coefficients whose determinant is only rounding noise is rejected.
- The zero matrix is singular.

This check does not measure conditioning: an invertible but very ill-conditioned matrix will yield an inaccurate inverse.

## Tests

```bash
cargo test                    # tests and doc tests, without dependencies
cargo test --features trig    # + rotations
```

The suite (`tests/matrix.rs`) checks known values, robustness (scales, NaN, ∞, zero matrix), and properties of deterministic pseudorandom matrices: `A·A⁻¹ ≈ I`, `det(AB) = det(A)·det(B)`, `(AB)ᵀ = BᵀAᵀ`, `solve`, and (with `trig`) rotation properties: determinant 1, orthogonality, composition, and the right-hand rule.

## Future ideas (not implemented)

Intentionally outside the current API, to keep the crate simple and predictable:

- **Arbitrary sizes**: stack-allocated `Matrix<const R: usize, const C: usize>` (alias `Matrix2x2 = Matrix<2, 2>`), with compile-time checked non-square matrix multiplication.
- **LU decomposition** with partial pivoting for `det`, `inv`, and `solve` at size n.
- **Cholesky** for symmetric positive-definite matrices (covariance matrices, Kalman filters).
- **`Vector<N>` type**: dot product, 3D cross product, norm (via `embedded-f32-sqrt`).
- **`#[must_use]`** on arithmetic operations.
- **Cycle benchmarks** (DWT counter on Cortex-M) and CI builds for embedded targets.

## 🛡️ License

Licensed under either of:

    MIT license (http://opensource.org/licenses/MIT)

    Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

at your option.

## 🦅 About

Developed by Jorge Andre Castro.