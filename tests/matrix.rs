//! Tests d'intégration de `embedded-matrix` (API publique uniquement).
//!
//! Lancer avec :  cargo test

use embedded_matrix::{Matrix2x2, Matrix3x3, MatrixError};

// ── Outils ───────────────────────────────────────────────────

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// Générateur pseudo-aléatoire déterministe, valeurs dans [-1, 1).
struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
    }

    /// Matrice 2×2 bien conditionnée (diagonale dominante).
    fn m2(&mut self) -> Matrix2x2 {
        let mut d = [[0.0f32; 2]; 2];
        for (i, row) in d.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = self.next() + if i == j { 2.5 } else { 0.0 };
            }
        }
        Matrix2x2::new(d)
    }

    /// Matrice 3×3 bien conditionnée (diagonale dominante).
    fn m3(&mut self) -> Matrix3x3 {
        let mut d = [[0.0f32; 3]; 3];
        for (i, row) in d.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = self.next() + if i == j { 4.0 } else { 0.0 };
            }
        }
        Matrix3x3::new(d)
    }
}

// ═════════════════════════════════════════════════════════════
//  Matrix2x2
// ═════════════════════════════════════════════════════════════

#[test]
fn m2_constructors() {
    assert_eq!(Matrix2x2::default(), Matrix2x2::zero());
    assert_eq!(Matrix2x2::from_diagonal([2.0, 3.0]), Matrix2x2::new([[2.0, 0.0], [0.0, 3.0]]));
    let m: Matrix2x2 = [[1.0, 2.0], [3.0, 4.0]].into();
    let a: [[f32; 2]; 2] = m.into();
    assert_eq!(a, [[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(m.as_array(), &a);
}

#[test]
fn m2_access_and_index() {
    let m = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(m.get(0, 1), 2.0);
    assert_eq!(m[(1, 0)], 3.0);
    assert_eq!(m[(1, 1)], 4.0);
}

#[test]
#[should_panic]
fn m2_index_out_of_bounds() {
    let m = Matrix2x2::identity();
    let _ = m[(2, 0)];
}

#[test]
fn m2_det_trace() {
    let m = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
    assert!(close(m.det(), -2.0, 1e-6));
    assert!(close(m.trace(), 5.0, 1e-6));
    assert!(close(Matrix2x2::identity().det(), 1.0, 1e-6));
}

#[test]
fn m2_arithmetic() {
    let a = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
    let b = Matrix2x2::new([[2.0, 0.0], [1.0, 3.0]]);

    assert_eq!(a + b, Matrix2x2::new([[3.0, 2.0], [4.0, 7.0]]));
    assert_eq!(a - b, Matrix2x2::new([[-1.0, 2.0], [2.0, 1.0]]));
    assert_eq!(a * b, Matrix2x2::new([[4.0, 6.0], [10.0, 12.0]]));
    assert_eq!(a * 3.0, Matrix2x2::new([[3.0, 6.0], [9.0, 12.0]]));
    assert_eq!(3.0 * a, a * 3.0);
    assert_eq!(-a, Matrix2x2::new([[-1.0, -2.0], [-3.0, -4.0]]));
    assert_eq!(a * Matrix2x2::identity(), a);

    // Les méthodes nommées donnent le même résultat que les opérateurs.
    assert_eq!(a.add(&b), a + b);
    assert_eq!(a.sub(&b), a - b);
    assert_eq!(a.mul(&b), a * b);
    assert_eq!(a.scale(3.0), a * 3.0);
}

#[test]
fn m2_assign_ops() {
    let a = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
    let b = Matrix2x2::identity();

    let mut m = a;
    m += b;
    assert_eq!(m, a + b);
    m -= b;
    assert_eq!(m, a);
    m *= 2.0;
    assert_eq!(m, a * 2.0);
    m = a;
    m *= b;
    assert_eq!(m, a);
}

#[test]
fn m2_transpose() {
    let m = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(m.transpose(), Matrix2x2::new([[1.0, 3.0], [2.0, 4.0]]));
    assert_eq!(m.transpose().transpose(), m);
}

#[test]
fn m2_mul_vec() {
    let m = Matrix2x2::new([[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(m.mul_vec([1.0, 1.0]), [3.0, 7.0]);
    assert_eq!(m * [1.0, 1.0], [3.0, 7.0]);
    assert_eq!(Matrix2x2::identity() * [5.0, -2.0], [5.0, -2.0]);
}

#[test]
fn m2_approx_eq() {
    let a = Matrix2x2::identity();
    let b = Matrix2x2::new([[1.0 + 1e-7, 0.0], [0.0, 1.0]]);
    assert!(a.approx_eq(&b, 1e-6));
    assert!(!a.approx_eq(&Matrix2x2::zero(), 1e-6));
    assert!(!a.approx_eq(&Matrix2x2::new([[f32::NAN, 0.0], [0.0, 1.0]]), 1.0));
}

#[test]
fn m2_inv_known() {
    let m = Matrix2x2::new([[4.0, 7.0], [2.0, 6.0]]); // det = 10
    let inv = m.inv().unwrap();
    assert!(inv.approx_eq(&Matrix2x2::new([[0.6, -0.7], [-0.2, 0.4]]), 1e-6));
}

#[test]
fn m2_inv_singular() {
    assert_eq!(Matrix2x2::new([[1.0, 2.0], [2.0, 4.0]]).inv(), Err(MatrixError::SingularMatrix));
    assert_eq!(Matrix2x2::zero().inv(), Err(MatrixError::SingularMatrix));
}

#[test]
fn m2_inv_non_finite() {
    assert_eq!(
        Matrix2x2::new([[f32::NAN, 0.0], [0.0, 1.0]]).inv(),
        Err(MatrixError::NonFiniteValue)
    );
    assert_eq!(
        Matrix2x2::new([[f32::INFINITY, 0.0], [0.0, 1.0]]).inv(),
        Err(MatrixError::NonFiniteValue)
    );
}

// ═════════════════════════════════════════════════════════════
//  Matrix3x3
// ═════════════════════════════════════════════════════════════

const M3_A: [[f32; 3]; 3] = [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];

#[test]
fn m3_constructors() {
    assert_eq!(Matrix3x3::default(), Matrix3x3::zero());
    assert_eq!(
        Matrix3x3::from_diagonal([1.0, 2.0, 3.0]),
        Matrix3x3::new([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    );
    let m: Matrix3x3 = M3_A.into();
    let a: [[f32; 3]; 3] = m.into();
    assert_eq!(a, M3_A);
    assert_eq!(m.as_array(), &M3_A);
}

#[test]
fn m3_access_and_index() {
    let m = Matrix3x3::new(M3_A);
    assert_eq!(m.get(0, 2), 3.0);
    assert_eq!(m[(2, 0)], 5.0);
    assert_eq!(m[(1, 2)], 4.0);
}

#[test]
#[should_panic]
fn m3_index_out_of_bounds() {
    let m = Matrix3x3::identity();
    let _ = m[(0, 3)];
}

#[test]
fn m3_det_trace() {
    assert!(close(Matrix3x3::identity().det(), 1.0, 1e-6));
    assert!(close(Matrix3x3::identity().trace(), 3.0, 1e-6));

    let tri = Matrix3x3::new([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]);
    assert!(close(tri.det(), 4.0, 1e-5));

    // det(M3_A) = 1*(0-24) - 2*(0-20) + 3*(0-5) = 1
    assert!(close(Matrix3x3::new(M3_A).det(), 1.0, 1e-5));
}

#[test]
fn m3_arithmetic() {
    let a = Matrix3x3::new(M3_A);
    let i = Matrix3x3::identity();

    assert_eq!(a * i, a);
    assert_eq!(i * a, a);
    assert_eq!((a + i) - i, a);
    assert_eq!(a * 2.0, a + a);
    assert_eq!(2.0 * a, a * 2.0);
    assert_eq!(-a + a, Matrix3x3::zero());

    assert_eq!(a.add(&i), a + i);
    assert_eq!(a.sub(&i), a - i);
    assert_eq!(a.mul(&i), a * i);
    assert_eq!(a.scale(2.0), a * 2.0);
}

#[test]
fn m3_mul_known() {
    let a = Matrix3x3::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let b = Matrix3x3::new([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);
    let want = Matrix3x3::new([[30.0, 24.0, 18.0], [84.0, 69.0, 54.0], [138.0, 114.0, 90.0]]);
    assert_eq!(a * b, want);
}

#[test]
fn m3_assign_ops() {
    let a = Matrix3x3::new(M3_A);
    let b = Matrix3x3::identity();

    let mut m = a;
    m += b;
    assert_eq!(m, a + b);
    m -= b;
    assert_eq!(m, a);
    m *= 2.0;
    assert_eq!(m, a * 2.0);
    m = a;
    m *= b;
    assert_eq!(m, a);
}

#[test]
fn m3_transpose() {
    let m = Matrix3x3::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    assert_eq!(
        m.transpose(),
        Matrix3x3::new([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])
    );
    assert_eq!(m.transpose().transpose(), m);
}

#[test]
fn m3_mul_vec() {
    let m = Matrix3x3::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    assert_eq!(m.mul_vec([1.0, 0.0, -1.0]), [-2.0, -2.0, -2.0]);
    assert_eq!(m * [1.0, 1.0, 1.0], [6.0, 15.0, 24.0]);
    assert_eq!(Matrix3x3::identity() * [5.0, -2.0, 7.0], [5.0, -2.0, 7.0]);
}

#[test]
fn m3_inv_known() {
    // Inverse exacte de M3_A (det = 1) : [[-24,18,5],[20,-15,-4],[-5,4,1]]
    let want = Matrix3x3::new([[-24.0, 18.0, 5.0], [20.0, -15.0, -4.0], [-5.0, 4.0, 1.0]]);
    let inv = Matrix3x3::new(M3_A).inv().unwrap();
    assert!(inv.approx_eq(&want, 1e-4), "{inv:?}");
}

#[test]
fn m3_inv_singular() {
    let m = Matrix3x3::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    assert_eq!(m.inv(), Err(MatrixError::SingularMatrix));
    assert_eq!(Matrix3x3::zero().inv(), Err(MatrixError::SingularMatrix));
    // Deux lignes identiques.
    let dup = Matrix3x3::new([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 0.0, 1.0]]);
    assert_eq!(dup.inv(), Err(MatrixError::SingularMatrix));
}

#[test]
fn m3_inv_non_finite() {
    let mut d = M3_A;
    d[1][1] = f32::NAN;
    assert_eq!(Matrix3x3::new(d).inv(), Err(MatrixError::NonFiniteValue));
    d[1][1] = f32::INFINITY;
    assert_eq!(Matrix3x3::new(d).inv(), Err(MatrixError::NonFiniteValue));
}

// ═════════════════════════════════════════════════════════════
//  Robustesse : seuil de singularité relatif à l'échelle
// ═════════════════════════════════════════════════════════════

#[test]
fn small_scale_matrices_stay_invertible() {
    // Un seuil absolu (|det| < 1e-7) les rejetterait à tort.
    let s2 = Matrix2x2::identity() * 1.0e-4;
    let inv2 = s2.inv().expect("1e-4·I est inversible");
    assert!(close(inv2.get(0, 0), 1.0e4, 1.0));

    let s3 = Matrix3x3::identity() * 1.0e-3;
    let inv3 = s3.inv().expect("1e-3·I est inversible");
    assert!(close(inv3.get(2, 2), 1.0e3, 1e-2));
}

#[test]
fn large_scale_matrices_invert() {
    let m = Matrix3x3::from_diagonal([1.0e3, 2.0e3, 4.0e3]);
    let inv = m.inv().unwrap();
    assert!((m * inv).approx_eq(&Matrix3x3::identity(), 1e-5));
}

#[test]
fn large_scale_singular_is_rejected() {
    // Lignes proportionnelles à grande échelle : det = bruit d'arrondi.
    let m = Matrix2x2::new([[1.0e4, 2.0e4], [3.0e4, 6.0e4]]);
    assert_eq!(m.inv(), Err(MatrixError::SingularMatrix));
}

// ═════════════════════════════════════════════════════════════
//  Propriétés (matrices pseudo-aléatoires, déterministes)
// ═════════════════════════════════════════════════════════════

#[test]
fn m2_inverse_roundtrip() {
    let mut rng = Lcg(42);
    for _ in 0..500 {
        let m = rng.m2();
        let inv = m.inv().unwrap();
        assert!((m * inv).approx_eq(&Matrix2x2::identity(), 1e-5), "{m:?}");
        assert!((inv * m).approx_eq(&Matrix2x2::identity(), 1e-5), "{m:?}");
    }
}

#[test]
fn m3_inverse_roundtrip() {
    let mut rng = Lcg(7);
    for _ in 0..500 {
        let m = rng.m3();
        let inv = m.inv().unwrap();
        assert!((m * inv).approx_eq(&Matrix3x3::identity(), 1e-5), "{m:?}");
        assert!((inv * m).approx_eq(&Matrix3x3::identity(), 1e-5), "{m:?}");
    }
}

#[test]
fn det_is_multiplicative() {
    let mut rng = Lcg(123);
    for _ in 0..300 {
        let (a, b) = (rng.m3(), rng.m3());
        let lhs = (a * b).det();
        let rhs = a.det() * b.det();
        assert!(close(lhs, rhs, 1e-4 * rhs.abs().max(1.0)), "{lhs} vs {rhs}");

        let (a, b) = (rng.m2(), rng.m2());
        let lhs = (a * b).det();
        let rhs = a.det() * b.det();
        assert!(close(lhs, rhs, 1e-4 * rhs.abs().max(1.0)), "{lhs} vs {rhs}");
    }
}

#[test]
fn transpose_of_product() {
    let mut rng = Lcg(99);
    for _ in 0..300 {
        let (a, b) = (rng.m3(), rng.m3());
        assert!((a * b).transpose().approx_eq(&(b.transpose() * a.transpose()), 1e-6));
        let (a, b) = (rng.m2(), rng.m2());
        assert!((a * b).transpose().approx_eq(&(b.transpose() * a.transpose()), 1e-6));
    }
}

#[test]
fn product_is_associative_within_tolerance() {
    let mut rng = Lcg(2026);
    for _ in 0..200 {
        let (a, b, c) = (rng.m3(), rng.m3(), rng.m3());
        assert!(((a * b) * c).approx_eq(&(a * (b * c)), 1e-3));
    }
}

#[test]
fn rotation_matrices_inverse_is_transpose() {
    // Matrices de rotation 2×2 : R⁻¹ = Rᵀ (std::f32::sin/cos côté test).
    for deg in 0..360 {
        let t = (deg as f32).to_radians();
        let (s, c) = t.sin_cos();
        let r = Matrix2x2::new([[c, -s], [s, c]]);
        assert!(r.inv().unwrap().approx_eq(&r.transpose(), 1e-5), "θ = {deg}°");
        assert!((r * r.transpose()).approx_eq(&Matrix2x2::identity(), 1e-5));
        assert!(close(r.det(), 1.0, 1e-5));
    }
}

#[test]
fn mul_vec_is_linear() {
    let mut rng = Lcg(5);
    for _ in 0..200 {
        let m = rng.m3();
        let u = [rng.next(), rng.next(), rng.next()];
        let v = [rng.next(), rng.next(), rng.next()];
        let sum = [u[0] + v[0], u[1] + v[1], u[2] + v[2]];

        let mu = m * u;
        let mv = m * v;
        let msum = m * sum;
        for k in 0..3 {
            assert!(close(msum[k], mu[k] + mv[k], 1e-5));
        }
    }
}

// ── Erreur ───────────────────────────────────────────────────

#[test]
fn error_display_is_not_empty() {
    assert!(!MatrixError::SingularMatrix.to_string().is_empty());
    assert!(!MatrixError::NonFiniteValue.to_string().is_empty());
}

// ═════════════════════════════════════════════════════════════
//  solve(b) : résolution de A·x = b
// ═════════════════════════════════════════════════════════════

#[test]
fn m2_solve_known() {
    let a = Matrix2x2::new([[2.0, 1.0], [1.0, 3.0]]);
    let x = a.solve([3.0, 5.0]).unwrap();
    assert!(close(x[0], 0.8, 1e-6) && close(x[1], 1.4, 1e-6), "{x:?}");
}

#[test]
fn m3_solve_known() {
    let a = Matrix3x3::new(M3_A);
    let x = a.solve([1.0, 2.0, 3.0]).unwrap();
    assert!(close(x[0], 27.0, 1e-3) && close(x[1], -22.0, 1e-3) && close(x[2], 6.0, 1e-3), "{x:?}");
}

#[test]
fn solve_errors() {
    assert_eq!(
        Matrix2x2::new([[1.0, 2.0], [2.0, 4.0]]).solve([1.0, 1.0]),
        Err(MatrixError::SingularMatrix)
    );
    assert_eq!(
        Matrix3x3::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).solve([1.0, 1.0, 1.0]),
        Err(MatrixError::SingularMatrix)
    );
    assert_eq!(
        Matrix2x2::identity().solve([f32::NAN, 0.0]),
        Err(MatrixError::NonFiniteValue)
    );
    assert_eq!(
        Matrix3x3::identity().solve([0.0, f32::INFINITY, 0.0]),
        Err(MatrixError::NonFiniteValue)
    );
}

#[test]
fn solve_roundtrip_and_matches_inverse() {
    let mut rng = Lcg(31337);
    for _ in 0..500 {
        let m = rng.m2();
        let b = [rng.next(), rng.next()];
        let x = m.solve(b).unwrap();
        let back = m * x;
        assert!(close(back[0], b[0], 1e-5) && close(back[1], b[1], 1e-5));
        let via_inv = m.inv().unwrap() * b;
        assert!(close(x[0], via_inv[0], 1e-5) && close(x[1], via_inv[1], 1e-5));

        let m = rng.m3();
        let b = [rng.next(), rng.next(), rng.next()];
        let x = m.solve(b).unwrap();
        let back = m * x;
        for k in 0..3 {
            assert!(close(back[k], b[k], 1e-5), "{m:?}");
        }
    }
}

// ═════════════════════════════════════════════════════════════
//  Rotations (feature `trig`) : cargo test --features trig
// ═════════════════════════════════════════════════════════════

#[cfg(feature = "trig")]
mod rotations {
    use super::*;
    use embedded_trig_f32::consts::{FRAC_PI_2, PI};

    #[test]
    fn rotation2_basic() {
        let p = Matrix2x2::rotation(FRAC_PI_2) * [1.0, 0.0];
        assert!(close(p[0], 0.0, 1e-6) && close(p[1], 1.0, 1e-6));
        assert!(Matrix2x2::rotation(0.0).approx_eq(&Matrix2x2::identity(), 1e-7));
        let half_turn = Matrix2x2::rotation(PI) * [1.0, 0.0];
        assert!(close(half_turn[0], -1.0, 1e-6) && close(half_turn[1], 0.0, 1e-6));
    }

    #[test]
    fn rotation2_properties() {
        for deg in (-720..=720).step_by(7) {
            let a = (deg as f32).to_radians();
            let r = Matrix2x2::rotation(a);
            assert!(close(r.det(), 1.0, 1e-5), "θ = {deg}°");
            assert!((r * r.transpose()).approx_eq(&Matrix2x2::identity(), 1e-5));
            assert!(r.inv().unwrap().approx_eq(&r.transpose(), 1e-5));
        }
    }

    #[test]
    fn rotation2_composition() {
        let (a, b) = (0.7f32, -1.9f32);
        let lhs = Matrix2x2::rotation(a) * Matrix2x2::rotation(b);
        assert!(lhs.approx_eq(&Matrix2x2::rotation(a + b), 1e-5));
    }

    #[test]
    fn rotation3_axes_fixed_and_orthonormal() {
        for deg in (-360..=360).step_by(11) {
            let a = (deg as f32).to_radians();
            let (rx, ry, rz) = (Matrix3x3::rotation_x(a), Matrix3x3::rotation_y(a), Matrix3x3::rotation_z(a));

            // L'axe de rotation reste invariant.
            assert_eq!(rx * [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
            assert_eq!(ry * [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]);
            assert_eq!(rz * [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]);

            for r in [rx, ry, rz] {
                assert!(close(r.det(), 1.0, 1e-5), "θ = {deg}°");
                assert!((r * r.transpose()).approx_eq(&Matrix3x3::identity(), 1e-5));
            }
        }
    }

    #[test]
    fn rotation3_right_hand_rule() {
        // Z : X → Y ; X : Y → Z ; Y : Z → X.
        let z = Matrix3x3::rotation_z(FRAC_PI_2) * [1.0, 0.0, 0.0];
        let x = Matrix3x3::rotation_x(FRAC_PI_2) * [0.0, 1.0, 0.0];
        let y = Matrix3x3::rotation_y(FRAC_PI_2) * [0.0, 0.0, 1.0];
        assert!(close(z[1], 1.0, 1e-6) && close(x[2], 1.0, 1e-6) && close(y[0], 1.0, 1e-6));
    }

    #[test]
    fn rotation3_z_matches_2d_block() {
        let a = 1.234f32;
        let (r2, r3) = (Matrix2x2::rotation(a), Matrix3x3::rotation_z(a));
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(r2.get(i, j), r3.get(i, j));
            }
        }
    }
}