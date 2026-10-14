/* The no-regression arm of nested_array_row_read_trace_fail: the same nondet
 * read out of a row, asserted over both stores. It cannot bite -- get()'s row
 * lowering runs only while a counterexample is built, and a SUCCESSFUL run
 * builds none -- so it guards the verdict rather than gating the fix. The
 * _fail twin is the one that aborts without it. */
int nondet_int(void);

int main(void)
{
  int a[2][2];
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);

  a[1][0] = 5;
  a[1][1] = 4;

  int v = a[1][i];
  __ESBMC_assert(v == 5 || v == 4, "the row read yields one of the row's stores");
  return 0;
}
