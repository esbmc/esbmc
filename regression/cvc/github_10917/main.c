/* esbmc/esbmc#10917: --all-witnesses reads array elements from the model
   inside its solver frame. */
int nondet_int();
int main()
{
  int a[3];
  for (int i = 0; i < 3; i++)
    a[i] = nondet_int();
  __ESBMC_assume(
    a[0] >= 0 && a[0] <= 5 && a[1] >= 0 && a[1] <= 5 && a[2] >= 0 && a[2] <= 5);
  __ESBMC_assert(a[0] + a[1] + a[2] != 5, "sum");
  return 0;
}
