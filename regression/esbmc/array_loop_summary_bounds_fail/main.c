// The summary keeps one iteration at a nondet index, so its bounds check
// still finds the symbolic overrun.
int a[50];
int nondet_int();
int main()
{
  int n = nondet_int();
  __ESBMC_assume(n <= 51);
  for (int i = 0; i < n; i++)
    a[i] = 1;
}
