// A local counter whose address is never taken is still unrolled, even with
// a call and a write through a pointer in the body: with no loop left, k=1
// proves the claim.
int g(int x)
{
  return x;
}

int main()
{
  int a[3];
  int *p = a;
  for (int i = 0; i < 3; i++)
    p[i] = g(i);
  __ESBMC_assert(a[2] == 2, "unrolled");
  return 0;
}
