// NumPy documentation: https://numpy.org/doc/stable/reference/routines.linalg.html

void dot(const int *A, int m, int n, const int *B, int n2, int p, int *C)
{
  int i = 0;
  int j = 0;
  int k = 0;

  while (i < m)
  {
    while (j < p)
    {
      C[i * p + j] = 0;
      while (k < n)
      {
        C[i * p + j] += A[i * n + k] * B[k * p + j];
        k++;
      }
      j++;
    }
    i++;
  }
}
