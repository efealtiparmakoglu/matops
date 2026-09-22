#include "matops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

mat_t mat_alloc(size_t rows, size_t cols) {
    mat_t m = { rows, cols, NULL };
    if (rows != 0 && cols != 0) {
        m.data = calloc(rows * cols, sizeof(double));
    }
    return m;
}

void mat_free(mat_t *m) {
    if (m && m->data) {
        free(m->data);
        m->data = NULL;
    }
}

mat_t mat_clone(const mat_t *m) {
    mat_t copy = mat_alloc(m->rows, m->cols);
    if (copy.data) {
        memcpy(copy.data, m->data, m->rows * m->cols * sizeof(double));
    }
    return copy;
}

/* Deterministic LCG so every run of the benchmark is reproducible. */
void mat_fill_random(mat_t *m, unsigned seed) {
    unsigned long long state = seed ? seed : 1u;
    size_t n = m->rows * m->cols;
    for (size_t i = 0; i < n; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        double v = (double)((state >> 33) & 0xFFFFF);
        m->data[i] = v / 524288.0 - 1.0; /* [-1, 1) */
    }
}

mat_t mat_identity(size_t n) {
    mat_t m = mat_alloc(n, n);
    for (size_t i = 0; i < n; i++) {
        mat_set(&m, i, i, 1.0);
    }
    return m;
}

double mat_get(const mat_t *m, size_t r, size_t c) {
    return m->data[r * m->cols + c];
}

void mat_set(mat_t *m, size_t r, size_t c, double v) {
    m->data[r * m->cols + c] = v;
}

mat_t mat_add(const mat_t *a, const mat_t *b) {
    mat_t out = mat_alloc(a->rows, a->cols);
    for (size_t i = 0; i < a->rows * a->cols; i++) {
        out.data[i] = a->data[i] + b->data[i];
    }
    return out;
}

mat_t mat_sub(const mat_t *a, const mat_t *b) {
    mat_t out = mat_alloc(a->rows, a->cols);
    for (size_t i = 0; i < a->rows * a->cols; i++) {
        out.data[i] = a->data[i] - b->data[i];
    }
    return out;
}

mat_t mat_transpose(const mat_t *a) {
    mat_t out = mat_alloc(a->cols, a->rows);
    for (size_t r = 0; r < a->rows; r++) {
        for (size_t c = 0; c < a->cols; c++) {
            out.data[c * a->rows + r] = mat_get(a, r, c);
        }
    }
    return out;
}

/* Multiplication kernels ------------------------------------------------
 *
 * The naive textbook loop is i,j,k: for every output element it walks a
 * column of b, hopping through memory with a stride of b->cols doubles.
 * Swapping the loops to i,k,j makes the inner loop stream both a's row
 * and b's row sequentially — same flop count, far fewer cache misses.
 */

mat_t mat_mul_naive(const mat_t *a, const mat_t *b) {
    mat_t out = mat_alloc(a->rows, b->cols);
    if (!out.data) return out;
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < a->cols; k++) {
                sum += mat_get(a, i, k) * mat_get(b, k, j);
            }
            out.data[i * b->cols + j] = sum;
        }
    }
    return out;
}

mat_t mat_mul_blocked(const mat_t *a, const mat_t *b) {
    mat_t out = mat_alloc(a->rows, b->cols);
    if (!out.data) return out;
    for (size_t i = 0; i < a->rows; i++) {
        const double *arow = a->data + i * a->cols;
        double *orow = out.data + i * b->cols;
        for (size_t k = 0; k < a->cols; k++) {
            double aik = arow[k];
            const double *brow = b->data + k * b->cols;
            for (size_t j = 0; j < b->cols; j++) {
                orow[j] += aik * brow[j];
            }
        }
    }
    return out;
}

/* --- Strassen -----------------------------------------------------------
 *
 * Classic 7-multiplication split. Inputs are padded up to a power of two
 * (block size <= STRASSEN_BASE falls back to the blocked kernel), so odd
 * sizes work without index gymnastics.
 */

static mat_t mat_mul_strassen_rec(const mat_t *a, const mat_t *b, size_t base) {
    size_t n = a->rows;
    if (n <= base) {
        return mat_mul_blocked(a, b);
    }
    size_t half = n / 2;

    mat_t a11 = mat_alloc(half, half), a12 = mat_alloc(half, half);
    mat_t a21 = mat_alloc(half, half), a22 = mat_alloc(half, half);
    mat_t b11 = mat_alloc(half, half), b12 = mat_alloc(half, half);
    mat_t b21 = mat_alloc(half, half), b22 = mat_alloc(half, half);
    for (size_t r = 0; r < half; r++) {
        for (size_t c = 0; c < half; c++) {
            mat_set(&a11, r, c, mat_get(a, r, c));
            mat_set(&a12, r, c, mat_get(a, r, c + half));
            mat_set(&a21, r, c, mat_get(a, r + half, c));
            mat_set(&a22, r, c, mat_get(a, r + half, c + half));
            mat_set(&b11, r, c, mat_get(b, r, c));
            mat_set(&b12, r, c, mat_get(b, r, c + half));
            mat_set(&b21, r, c, mat_get(b, r + half, c));
            mat_set(&b22, r, c, mat_get(b, r + half, c + half));
        }
    }

    mat_t s1 = mat_sub(&b12, &b22);
    mat_t s2 = mat_add(&a11, &a12);
    mat_t s3 = mat_add(&a21, &a22);
    mat_t s4 = mat_sub(&b21, &b11);
    mat_t s5 = mat_add(&a11, &a22);
    mat_t s6 = mat_add(&b11, &b22);
    mat_t s7 = mat_sub(&a12, &a22);
    mat_t s8 = mat_add(&b21, &b22);
    mat_t s9 = mat_sub(&a11, &a21);
    mat_t s10 = mat_add(&b11, &b12);

    mat_t m1 = mat_mul_strassen_rec(&a11, &s1, base);
    mat_t m2 = mat_mul_strassen_rec(&s2, &b22, base);
    mat_t m3 = mat_mul_strassen_rec(&s3, &b11, base);
    mat_t m4 = mat_mul_strassen_rec(&a22, &s4, base);
    mat_t m5 = mat_mul_strassen_rec(&s5, &s6, base);
    mat_t m6 = mat_mul_strassen_rec(&s7, &s8, base);
    mat_t m7 = mat_mul_strassen_rec(&s9, &s10, base);

    mat_t m5_plus_m4 = mat_add(&m5, &m4);
    mat_t m5p4_minus_m2 = mat_sub(&m5_plus_m4, &m2);
    mat_t c11 = mat_add(&m5p4_minus_m2, &m6);
    mat_t c12 = mat_add(&m1, &m2);
    mat_t c21 = mat_add(&m3, &m4);
    mat_t m5_plus_m1 = mat_add(&m5, &m1);
    mat_t m5p1_minus_m3 = mat_sub(&m5_plus_m1, &m3);
    mat_t c22 = mat_sub(&m5p1_minus_m3, &m7);

    mat_t out = mat_alloc(n, n);
    for (size_t r = 0; r < half; r++) {
        for (size_t c = 0; c < half; c++) {
            mat_set(&out, r, c, mat_get(&c11, r, c));
            mat_set(&out, r, c + half, mat_get(&c12, r, c));
            mat_set(&out, r + half, c, mat_get(&c21, r, c));
            mat_set(&out, r + half, c + half, mat_get(&c22, r, c));
        }
    }

    mat_free(&a11); mat_free(&a12); mat_free(&a21); mat_free(&a22);
    mat_free(&b11); mat_free(&b12); mat_free(&b21); mat_free(&b22);
    mat_free(&s1); mat_free(&s2); mat_free(&s3); mat_free(&s4);
    mat_free(&s5); mat_free(&s6); mat_free(&s7); mat_free(&s8);
    mat_free(&s9); mat_free(&s10);
    mat_free(&m1); mat_free(&m2); mat_free(&m3); mat_free(&m4);
    mat_free(&m5); mat_free(&m6); mat_free(&m7);
    mat_free(&m5_plus_m4); mat_free(&m5p4_minus_m2);
    mat_free(&m5_plus_m1); mat_free(&m5p1_minus_m3);
    mat_free(&c11); mat_free(&c12); mat_free(&c21); mat_free(&c22);
    return out;
}

static size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

mat_t mat_mul_strassen(const mat_t *a, const mat_t *b) {
    /* pad every dimension, including the shared inner one, to a power of two */
    size_t biggest = a->rows;
    if (a->cols > biggest) biggest = a->cols;
    if (b->cols > biggest) biggest = b->cols;
    size_t n = next_pow2(biggest);
    if (n < 2) n = 2;
    mat_t pa = mat_alloc(n, n), pb = mat_alloc(n, n);
    for (size_t r = 0; r < a->rows; r++) {
        memcpy(pa.data + r * n, a->data + r * a->cols, a->cols * sizeof(double));
    }
    for (size_t r = 0; r < b->rows; r++) {
        memcpy(pb.data + r * n, b->data + r * b->cols, b->cols * sizeof(double));
    }
    mat_t full = mat_mul_strassen_rec(&pa, &pb, 64);

    /* slice the useful quadrant back out */
    mat_t out = mat_alloc(a->rows, b->cols);
    for (size_t r = 0; r < a->rows; r++) {
        memcpy(out.data + r * b->cols, full.data + r * n, b->cols * sizeof(double));
    }
    mat_free(&pa);
    mat_free(&pb);
    mat_free(&full);
    return out;
}

/* LU with partial pivoting ---------------------------------------------- */

int mat_lu_decompose(const mat_t *a, mat_t *lu, size_t *perm, double *sign) {
    size_t n = a->rows;
    *lu = mat_clone(a);
    for (size_t i = 0; i < n; i++) perm[i] = i;
    *sign = 1.0;

    for (size_t col = 0; col < n; col++) {
        /* pick the largest pivot in this column */
        size_t pivot = col;
        double best = fabs(mat_get(lu, col, col));
        for (size_t r = col + 1; r < n; r++) {
            double v = fabs(mat_get(lu, r, col));
            if (v > best) {
                best = v;
                pivot = r;
            }
        }
        if (best == 0.0) {
            return -1; /* singular */
        }
        if (pivot != col) {
            for (size_t c = 0; c < n; c++) {
                double tmp = mat_get(lu, col, c);
                mat_set(lu, col, c, mat_get(lu, pivot, c));
                mat_set(lu, pivot, c, tmp);
            }
            size_t t = perm[col];
            perm[col] = perm[pivot];
            perm[pivot] = t;
            *sign = -*sign;
        }
        double pivot_value = mat_get(lu, col, col);
        for (size_t r = col + 1; r < n; r++) {
            double factor = mat_get(lu, r, col) / pivot_value;
            mat_set(lu, r, col, factor);
            for (size_t c = col + 1; c < n; c++) {
                double v = mat_get(lu, r, c) - factor * mat_get(lu, col, c);
                mat_set(lu, r, c, v);
            }
        }
    }
    return 0;
}

double mat_det(const mat_t *a, int *ok) {
    if (a->rows != a->cols) {
        *ok = 0;
        return 0.0;
    }
    *ok = 1;
    if (a->rows == 0) return 1.0;
    mat_t lu;
    size_t *perm = malloc(a->rows * sizeof(size_t));
    double sign = 1.0;
    int singular = mat_lu_decompose(a, &lu, perm, &sign);
    double det = singular == 0 ? sign : 0.0;
    if (singular == 0) {
        for (size_t i = 0; i < a->rows; i++) {
            det *= mat_get(&lu, i, i);
        }
    }
    mat_free(&lu);
    free(perm);
    return det;
}

mat_t mat_solve(const mat_t *a, const mat_t *b, int *ok) {
    size_t n = a->rows;
    mat_t x = mat_alloc(n, b->cols);
    *ok = 1;
    if (a->rows != a->cols) {
        *ok = 0;
        return x;
    }
    mat_t lu;
    size_t *perm = malloc(n * sizeof(size_t));
    double sign = 1.0;
    if (mat_lu_decompose(a, &lu, perm, &sign) != 0) {
        *ok = 0;
        mat_free(&lu);
        free(perm);
        return x;
    }

    /* solve L y = P b (forward), then U x = y (backward), per column of b */
    mat_t pb = mat_alloc(n, b->cols);
    for (size_t r = 0; r < n; r++) {
        for (size_t c = 0; c < b->cols; c++) {
            mat_set(&pb, r, c, mat_get(b, perm[r], c));
        }
    }
    mat_t y = mat_alloc(n, b->cols);
    for (size_t c = 0; c < b->cols; c++) {
        for (size_t r = 0; r < n; r++) {
            double sum = mat_get(&pb, r, c);
            for (size_t k = 0; k < r; k++) {
                sum -= mat_get(&lu, r, k) * mat_get(&y, k, c);
            }
            mat_set(&y, r, c, sum); /* L has implicit unit diagonal */
        }
    }
    for (size_t c = 0; c < b->cols; c++) {
        for (size_t ii = n; ii-- > 0;) {
            double sum = mat_get(&y, ii, c);
            for (size_t k = ii + 1; k < n; k++) {
                sum -= mat_get(&lu, ii, k) * mat_get(&x, k, c);
            }
            mat_set(&x, ii, c, sum / mat_get(&lu, ii, ii));
        }
    }

    mat_free(&lu);
    mat_free(&pb);
    mat_free(&y);
    free(perm);
    return x;
}

double mat_max_abs_diff(const mat_t *a, const mat_t *b) {
    if (a->rows != b->rows || a->cols != b->cols) return INFINITY;
    double max = 0.0;
    for (size_t i = 0; i < a->rows * a->cols; i++) {
        double d = fabs(a->data[i] - b->data[i]);
        if (d > max) max = d;
    }
    return max;
}

void mat_print(const mat_t *m) {
    for (size_t r = 0; r < m->rows; r++) {
        for (size_t c = 0; c < m->cols; c++) {
            printf("%10.4f ", mat_get(m, r, c));
        }
        putchar('\n');
    }
}
