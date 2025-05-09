// NumPy documentation: https://numpy.org/doc/stable/reference/routines.linalg.html

#include <assert.h>
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
