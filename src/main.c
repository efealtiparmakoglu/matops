/* matops CLI: benchmark the three multiply kernels, verify they agree,
 * and demo determinant / solve. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "matops.h"

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void usage(void) {
    fprintf(stderr,
            "matops v1.0.0 — dense matrix library without BLAS\n\n"
            "USAGE\n"
            "  matops bench [-n N]        multiply kernels: naive vs ikj vs strassen\n"
            "  matops det [-n N]          determinant of a random n x n matrix\n"
            "  matops solve [-n N]        solve A x = b and check the residual\n\n"
            "Exit codes: 0 ok, 1 failure, 2 usage error\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage();
        return 2;
    }
    size_t n = 256;
    for (int i = 2; i < argc - 1; i++) {
        if (strcmp(argv[i], "-n") == 0) {
            n = (size_t)strtoul(argv[i + 1], NULL, 10);
            i++;
        }
    }

    if (strcmp(argv[1], "bench") == 0) {
        mat_t a = mat_alloc(n, n), b = mat_alloc(n, n);
        mat_fill_random(&a, 1);
        mat_fill_random(&b, 2);

        double t0 = now_seconds();
        mat_t naive = mat_mul_naive(&a, &b);
        double t_naive = now_seconds() - t0;

        t0 = now_seconds();
        mat_t ikj = mat_mul_blocked(&a, &b);
        double t_ikj = now_seconds() - t0;

        t0 = now_seconds();
        mat_t strassen = mat_mul_strassen(&a, &b);
        double t_strassen = now_seconds() - t0;

        double diff_ikj = mat_max_abs_diff(&naive, &ikj);
        double diff_strassen = mat_max_abs_diff(&naive, &strassen);
        int agree = diff_ikj < 1e-6 && diff_strassen < 1e-4;

        printf("n = %zu\n", n);
        printf("naive    ijk: %8.1f ms\n", t_naive * 1000.0);
        printf("blocked  ikj: %8.1f ms  (%.2fx vs naive)\n", t_ikj * 1000.0, t_naive / t_ikj);
        printf("strassen     : %8.1f ms  (%.2fx vs naive)\n", t_strassen * 1000.0,
               t_naive / t_strassen);
        printf("agreement: ikj %.2e, strassen %.2e -> %s\n", diff_ikj, diff_strassen,
               agree ? "OK" : "MISMATCH");
        printf("flops: %.1f M per multiply\n", 2.0 * (double)n * (double)n * (double)n / 1e6);

        mat_free(&a); mat_free(&b); mat_free(&naive); mat_free(&ikj); mat_free(&strassen);
        return agree ? 0 : 1;
    }

    if (strcmp(argv[1], "det") == 0) {
        mat_t a = mat_alloc(n, n);
        mat_fill_random(&a, 7);
        int ok = 0;
        double t0 = now_seconds();
        double det = mat_det(&a, &ok);
        double elapsed = now_seconds() - t0;
        printf("det(random %zu x %zu) = %.6e  (%.1f ms, ok=%d)\n", n, n, det, elapsed * 1000.0, ok);
        mat_free(&a);
        return ok ? 0 : 1;
    }

    if (strcmp(argv[1], "solve") == 0) {
        mat_t a = mat_alloc(n, n), x_true = mat_alloc(n, 1);
        mat_fill_random(&a, 3);
        mat_fill_random(&x_true, 4);
        for (size_t r = 0; r < n; r++) {
            /* diagonal dominance so the system is non-singular */
            mat_set(&a, r, r, mat_get(&a, r, r) + (double)n);
        }
        mat_t b = mat_mul_blocked(&a, &x_true);
        int ok = 0;
        mat_t x = mat_solve(&a, &b, &ok);
        mat_t residual = mat_mul_blocked(&a, &x);
        mat_t diff = mat_sub(&residual, &b);
        double max_error = 0.0;
        for (size_t i = 0; i < n; i++) {
            double v = fabs(diff.data[i]);
            if (v > max_error) max_error = v;
        }
        printf("solve %zu x %zu: residual max = %.2e (%s)\n", n, n, max_error,
               ok && max_error < 1e-6 ? "OK" : "FAIL");
        mat_free(&a); mat_free(&x_true); mat_free(&b); mat_free(&x); mat_free(&residual);
        mat_free(&diff);
        return ok && max_error < 1e-6 ? 0 : 1;
    }

    usage();
    return 2;
}
