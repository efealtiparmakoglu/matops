# matops

A dense matrix library written **from scratch** — no BLAS, no LAPACK,
just C11: three multiply kernels, LU decomposition with partial
pivoting, determinant and a solver, with numerical tests.

```text
$ matops bench -n 256
n = 256
naive    ijk:     14.1 ms
blocked  ikj:      2.4 ms  (5.90x vs naive)
strassen     :      2.3 ms  (6.03x vs naive)
agreement: ikj 0.00e+00, strassen 0.00e+00 -> OK
```

## The interesting parts

- **Cache-friendly multiply** — the textbook `i,j,k` loop walks columns
  of B with a stride of `b->cols` doubles; reordering to `i,k,j` streams
  both operand rows sequentially. Same flop count, ~6x faster at
  n = 256 — the benchmark prints the measured gap.
- **Strassen** — classic 7-multiplication recursion, inputs padded to a
  power of two (every dimension, including the shared inner one),
  base case 64 falls through to the blocked kernel. Accepts rectangular
  inputs and slices the useful quadrant back out.
- **LU with partial pivoting** — `P·A = L·U` with the permutation and
  sign tracked; determinant is the signed product of U's diagonal, the
  solver does forward/back substitution. Numerical tests reconstruct
  `P·A` from `L·U` and check `det(A·B) = det(A)·det(B)`.

## CLI

```
matops bench [-n N]   benchmark + cross-verify the three kernels
matops det   [-n N]   determinant of a random matrix
matops solve [-n N]   solve A x = b, report the max residual
```

## Tests

```console
$ make test     # 27 checks: kernels agree (incl. rectangular Strassen),
                # LU reconstruction, solve residual, known determinants,
                # det(A·B) = det(A)det(B)
```

Strassen's tolerance is 1e-6 rather than 1e-8 on purpose — its extra
matrix additions are the documented price of the saved multiplication.

## License

MIT — see [LICENSE](LICENSE).
