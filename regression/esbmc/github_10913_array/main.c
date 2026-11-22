/* esbmc/esbmc#10913: Boolector enumeration with blocking clauses over
   array-element inputs. */
int nondet_int();
int main()
{
  int a[3];
  for (int i = 0; i < 3; i++)
    a[i] = nondet_int();
  __ESBMC_assume(a[0] >= 0 && a[1] >= 0 && a[2] >= 0);
  __ESBMC_assert(a[0] + a[1] + a[2] != 5, "sum");
  return 0;
}
