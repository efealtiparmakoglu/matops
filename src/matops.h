/* matops.h — small dense matrix library written from scratch (no BLAS).
 *
 * Row-major doubles. Three multiply kernels (naive ijk, cache-friendly
 * ikj, Strassen), LU decomposition with partial pivoting, determinant
 * and a solver, all unit-tested numerically.
 */
#ifndef MATOPS_H
#define MATOPS_H

#include <stddef.h>

typedef struct {
    size_t rows, cols;
    double *data;
} mat_t;

/* lifecycle */
mat_t mat_alloc(size_t rows, size_t cols);
void mat_free(mat_t *m);
mat_t mat_clone(const mat_t *m);
void mat_fill_random(mat_t *m, unsigned seed); /* deterministic LCG */
mat_t mat_identity(size_t n);

/* element access */
double mat_get(const mat_t *m, size_t r, size_t c);
void mat_set(mat_t *m, size_t r, size_t c, double v);

/* arithmetic */
mat_t mat_add(const mat_t *a, const mat_t *b);
mat_t mat_sub(const mat_t *a, const mat_t *b);
mat_t mat_transpose(const mat_t *a);

/* multiplication kernels */
mat_t mat_mul_naive(const mat_t *a, const mat_t *b);    /* i,j,k order */
mat_t mat_mul_blocked(const mat_t *a, const mat_t *b);  /* i,k,j order (cache friendly) */
mat_t mat_mul_strassen(const mat_t *a, const mat_t *b); /* recursive, padded */

/* linear algebra */
/* LU with partial pivoting: P A = L U, stored in lu (L lower, unit diag
 * implicit) and U upper. perm[out] receives the row permutation applied to
 * A (perm[i] = original row now at i), sign[out] the permutation sign.
 * Returns 0 on success, -1 if a is singular (decomposition still written). */
int mat_lu_decompose(const mat_t *a, mat_t *lu, size_t *perm, double *sign);
double mat_det(const mat_t *a, int *ok);
/* Solve A x = b via LU; b has `rows` lines of `cols` columns. */
mat_t mat_solve(const mat_t *a, const mat_t *b, int *ok);

/* helpers */
double mat_max_abs_diff(const mat_t *a, const mat_t *b);
void mat_print(const mat_t *m);

#endif /* MATOPS_H */
