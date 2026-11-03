// esbmc/esbmc#8681: a guarded index into the delete operand passes its checks.
int nondet_int();
int main()
{
  int *a[2] = {nullptr, new int};
  int *b[2] = {new int[2], nullptr};
  int k = nondet_int();
  __ESBMC_assume(k == 1 || k == 2);
  delete a[2 / (k + 1)];
  delete[] b[(k + 1) / 4];
  return 0;
}
