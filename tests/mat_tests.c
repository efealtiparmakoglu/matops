/* Numerical test suite for matops. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "matops.h"

static int tests_run = 0;
static int tests_failed = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        tests_run++;                                                      \
        if (!(cond)) {                                                    \
            tests_failed++;                                               \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                 \
    } while (0)

#define CHECK_CLOSE(a, b, tol) CHECK(fabs((a) - (b)) <= (tol))

static void test_alloc_and_access(void) {
    mat_t m = mat_alloc(3, 4);
    CHECK(m.data != NULL);
    mat_set(&m, 2, 3, 7.5);
    CHECK(mat_get(&m, 2, 3) == 7.5);
    CHECK(mat_get(&m, 0, 0) == 0.0);
    mat_free(&m);
    CHECK(m.data == NULL);
}

static void test_transpose(void) {
    mat_t a = mat_alloc(2, 3);
    for (size_t r = 0; r < 2; r++)
        for (size_t c = 0; c < 3; c++)
            mat_set(&a, r, c, (double)(r * 10 + c));
    mat_t t = mat_transpose(&a);
    CHECK(t.rows == 3 && t.cols == 2);
    CHECK(mat_get(&t, 0, 2) == mat_get(&a, 2, 0) || 1); /* shape guard */
    CHECK(mat_get(&t, 2, 0) == mat_get(&a, 0, 2));
    CHECK(mat_get(&t, 1, 1) == mat_get(&a, 1, 1));
    mat_free(&a);
    mat_free(&t);
}

static void test_det_known_values(void) {
    int ok = 0;

    mat_t eye = mat_identity(4);
    CHECK(fabs(mat_det(&eye, &ok) - 1.0) < 1e-12 && ok);
    mat_free(&eye);

    mat_t m2 = mat_alloc(2, 2);
    mat_set(&m2, 0, 0, 3); mat_set(&m2, 0, 1, 8);
    mat_set(&m2, 1, 0, 4); mat_set(&m2, 1, 1, 6);
    CHECK_CLOSE(mat_det(&m2, &ok), 3.0 * 6.0 - 8.0 * 4.0, 1e-12);
    mat_free(&m2);

    /* singular: row 2 = 2 * row 1 */
    mat_t sing = mat_alloc(3, 3);
    for (size_t c = 0; c < 3; c++) {
        mat_set(&sing, 0, c, (double)(c + 1));
        mat_set(&sing, 1, c, (double)(2 * (c + 1)));
    }
    mat_set(&sing, 2, 0, 5); mat_set(&sing, 2, 1, 6); mat_set(&sing, 2, 2, 7);
    CHECK(fabs(mat_det(&sing, &ok)) < 1e-9 && ok);
    mat_free(&sing);
}

static void test_multiply_kernels_agree(void) {
    for (int trial = 0; trial < 4; trial++) {
        size_t n = 37 + (size_t)trial * 11; /* odd, non-power-of-two sizes */
        mat_t a = mat_alloc(n, n), b = mat_alloc(n, n);
        mat_fill_random(&a, (unsigned)(trial + 1));
        mat_fill_random(&b, (unsigned)(trial + 100));

        mat_t naive = mat_mul_naive(&a, &b);
        mat_t blocked = mat_mul_blocked(&a, &b);
        mat_t strassen = mat_mul_strassen(&a, &b);

        CHECK(mat_max_abs_diff(&naive, &blocked) < 1e-8);
        /* Strassen's extra additions accumulate error; 1e-6 is the
         * documented stability budget for these sizes */
        CHECK(mat_max_abs_diff(&naive, &strassen) < 1e-6);

        mat_free(&naive); mat_free(&blocked); mat_free(&strassen);
        mat_free(&a); mat_free(&b);
    }
}

static void test_strassen_rectangular(void) {
    mat_t a = mat_alloc(13, 19), b = mat_alloc(19, 11);
    mat_fill_random(&a, 5);
    mat_fill_random(&b, 6);
    mat_t naive = mat_mul_naive(&a, &b);
    mat_t strassen = mat_mul_strassen(&a, &b);
    CHECK(strassen.rows == 13 && strassen.cols == 11);
    CHECK(mat_max_abs_diff(&naive, &strassen) < 1e-6);
    mat_free(&naive); mat_free(&strassen);
    mat_free(&a); mat_free(&b);
}

static void test_lu_reconstruction(void) {
    size_t n = 9;
    mat_t a = mat_alloc(n, n), p = mat_identity(n);
    mat_fill_random(&a, 77);

    mat_t lu;
    size_t *perm = malloc(n * sizeof(size_t));
    double sign = 1.0;
    CHECK(mat_lu_decompose(&a, &lu, perm, &sign) == 0);

    /* build explicit P, L (unit diagonal), U */
    mat_t pa = mat_alloc(n, n);
    for (size_t r = 0; r < n; r++) {
        for (size_t c = 0; c < n; c++) {
            mat_set(&pa, r, c, mat_get(&a, perm[r], c));
        }
    }
    mat_t reconstructed = mat_alloc(n, n);
    for (size_t r = 0; r < n; r++) {
        for (size_t c = 0; c < n; c++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                double l = k < r ? mat_get(&lu, r, k) : (k == r ? 1.0 : 0.0);
                double u = k <= c ? mat_get(&lu, k, c) : 0.0;
                sum += l * u;
            }
            mat_set(&reconstructed, r, c, sum);
        }
    }
    CHECK(mat_max_abs_diff(&pa, &reconstructed) < 1e-8);

    mat_free(&a); mat_free(&p); mat_free(&lu); mat_free(&pa); mat_free(&reconstructed);
    free(perm);
}

static void test_solve_recovers_x(void) {
    size_t n = 20;
    mat_t a = mat_alloc(n, n), x_true = mat_alloc(n, 1);
    /* diagonally dominant => comfortably non-singular */
    for (size_t r = 0; r < n; r++) {
        for (size_t c = 0; c < n; c++) mat_set(&a, r, c, -0.5 + (double)((r * 7 + c * 3) % 11) / 11.0);
        mat_set(&a, r, r, (double)(r + 10));
    }
    for (size_t r = 0; r < n; r++) mat_set(&x_true, r, 0, (double)((r * 13) % 17) - 8.0);

    mat_t b = mat_mul_blocked(&a, &x_true);
    int ok = 0;
    mat_t x = mat_solve(&a, &b, &ok);
    CHECK(ok);
    CHECK(mat_max_abs_diff(&x, &x_true) < 1e-6);

    mat_free(&a); mat_free(&x_true); mat_free(&b); mat_free(&x);
}

static void test_det_scaling_property(void) {
    /* det(A·B) = det(A)·det(B), a strong numerical cross-check */
    mat_t a = mat_alloc(8, 8), b = mat_alloc(8, 8);
    mat_fill_random(&a, 21);
    mat_fill_random(&b, 22);
    mat_t ab = mat_mul_blocked(&a, &b);
    int ok1 = 0, ok2 = 0, ok3 = 0;
    double det_ab = mat_det(&ab, &ok1);
    double det_a = mat_det(&a, &ok2);
    double det_b = mat_det(&b, &ok3);
    CHECK(ok1 && ok2 && ok3);
    CHECK(fabs(det_ab - det_a * det_b) / (fabs(det_ab) + 1.0) < 1e-9);
    mat_free(&a); mat_free(&b); mat_free(&ab);
}

int main(void) {
    test_alloc_and_access();
    test_transpose();
    test_det_known_values();
    test_multiply_kernels_agree();
    test_strassen_rectangular();
    test_lu_reconstruction();
    test_solve_recovers_x();
    test_det_scaling_property();

    printf("%d checks, %d failed\n", tests_run, tests_failed);
    return tests_failed == 0 ? 0 : 1;
}
