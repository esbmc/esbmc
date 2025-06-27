// NumPy documentation: https://numpy.org/doc/stable/reference/routines.linalg.html
#include <math.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>

void dot(int64_t *A, int64_t *B, int64_t *C, int64_t m, int64_t n, int64_t p)
{
  int i = 0;
  while (i < m)
  {
    int j = 0;
    while (j < p)
    {
      int sum = 0;
      int k = 0;
      while (k < n)
      {
        sum += *(A + i * n + k) * *(B + k * p + j);
        k++;
      }
      *(C + i * p + j) = sum;
      j++;
    }
    i++;
  }
}

void matmul(int64_t *A, int64_t *B, int64_t *C, int64_t m, int64_t n, int64_t p)
{
  dot(A, B, C, m, n, p);
}

void matmul(int64_t *A, int64_t *B, int64_t *C, int64_t m, int64_t n, int64_t p)
{
  dot(A, B, C, m, n, p);
}

void transpose(int64_t *src, int64_t *dst, int64_t rows, int64_t cols)
{
  int i = 0;
  while (i < rows)
  {
    int j = 0;
    while (j < cols)
    {
      dst[j * rows + i] = src[i * cols + j];
      ++j;
    }
    ++i;
  }
}

#if 0
#  define IDX(i, j, n) ((i) * (n) + (j))

void det(const double *src, double *dst, int n)
{
  double *mat = __ESBMC_alloca(n * n * sizeof(double));

  //  int i = 0;
  //  while (i < n * n)
  //  {
  //    mat[i] = src[i];
  //    i++;
  //  }

  memcpy(mat, src, n * n * sizeof(double));

  double det = 1.0;
  int k = 0;

  while (k < n)
  {
    double pivot = mat[IDX(k, k, n)];

    if (fabs(pivot) < 1e-12)
    {
      det = 0.0;
      break;
    }

    int i2 = k + 1;
    while (i2 < n)
    {
      double factor = mat[IDX(i2, k, n)] / pivot;
      int j = k;
      while (j < n)
      {
        mat[IDX(i2, j, n)] -= factor * mat[IDX(k, j, n)];
        j++;
      }
      i2++;
    }

    det *= pivot;
    k++;
  }

  *dst = det;
}
#endif

void det(const int64_t *src, int64_t *dst, int64_t rows, int64_t cols)
{
  int64_t a = src[0 * cols + 0];
  int64_t b = src[0 * cols + 1];
  int64_t c = src[1 * cols + 0];
  int64_t d = src[1 * cols + 1];

  *dst = a * d - b * c;
}

