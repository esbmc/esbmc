#include <stdint.h>

void add(int64_t *A, int64_t *B, int64_t *C, int64_t m, int64_t n)
{
  int i = 0;
  while (i < m)
  {
    int j = 0;
    while (j < n)
    {
      *(C + i * n + j) = *(A + i * n + j) + *(B + i * n + j);
      j++;
    }
    i++;
  }
}

void subtract(int64_t *A, int64_t *B, int64_t *C, int64_t m, int64_t n)
{
  int i = 0;
  while (i < m)
  {
    int j = 0;
    while (j < n)
    {
      *(C + i * n + j) = *(A + i * n + j) - *(B + i * n + j);
      j++;
    }
    i++;
  }
}

void multiply(int64_t *A, int64_t *B, int64_t *C, int64_t m, int64_t n)
{
  int i = 0;
  while (i < m)
  {
    int j = 0;
    while (j < n)
    {
      *(C + i * n + j) = *(A + i * n + j) * *(B + i * n + j);
      j++;
    }
    i++;
  }
}

void divide(int64_t *A, int64_t *B, int64_t *C, int64_t m, int64_t n)
{
  int i = 0;
  while (i < m)
  {
    int j = 0;
    while (j < n)
    {
      *(C + i * n + j) = *(A + i * n + j) / *(B + i * n + j);
      j++;
    }
    i++;
  }
}
