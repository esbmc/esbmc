#include <stdint.h>
int main()
{
  unsigned N, M;
  __ESBMC_assume(N > 30);
  __ESBMC_assume(M > 30);
  __ESBMC_assume(N < SIZE_MAX / sizeof(int) / M);
  int arr[N][M];
  arr[15][25] = arr[10][15];
  // while(1)
  {
    __ESBMC_assert(arr[15][25] == arr[10][15], "This should be constant");
  }

  return 0;
}
