// __ESBMC_unroll(2) stops the loop by a[2]; a summary would write a[3..7]
// out of bounds.
int main()
{
  int a[3] = {0};
  __ESBMC_unroll(2);
  for (unsigned j = 0; j < 8; j++)
    a[j] = (int)(j + 1);
  return 0;
}
