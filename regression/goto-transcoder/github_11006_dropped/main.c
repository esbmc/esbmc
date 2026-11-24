/* Invariants ESBMC cannot express are dropped and the loops unwound. */
int a[4];
int main()
{
  int *p = a;
  for (unsigned i = 0; i < 4; i++)
    __CPROVER_loop_invariant(i <= 4 && __CPROVER_pointer_in_range(a, p, a + 4))
  {
    a[i] = 0;
  }
  for (unsigned i = 0; i < 4; i++)
    __CPROVER_loop_invariant(i <= 4 && __CPROVER_r_ok(p, 4 * sizeof(int)))
  {
    a[i] = 1;
  }
  for (unsigned i = 0; i < 4; i++)
    __CPROVER_loop_invariant(i <= 4 && __CPROVER_DYNAMIC_OBJECT(p) == 0)
  {
    a[i] = 2;
  }
  __CPROVER_assert(a[2] == 2, "last loop wrote a[2]");
  return 0;
}
