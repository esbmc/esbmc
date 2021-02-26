int main()
{
  unsigned N, M, O;
  __ESBMC_assume(N > 30);
  __ESBMC_assume(M > 30);
  __ESBMC_assume(O > 10);
  int arr[N][M][O];
  arr[15][25][1] = arr[10][15][2];
  while(1)
  {
    __ESBMC_assert(arr[15][25][2] == arr[10][15][2], "This should be constant");
  }

  return 0;
}
