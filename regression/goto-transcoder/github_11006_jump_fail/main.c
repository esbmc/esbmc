/* The goto enters the loop past the invariant's base case. */
unsigned nondet_uint(void);
_Bool nondet_bool(void);
int main()
{
  unsigned n = nondet_uint(), i = 0, k = 0;
  __CPROVER_assume(n >= 1 && n <= 3);
  if (nondet_bool())
  {
    k = 1;
    goto L;
  }
L:
  while (i < n)
    __CPROVER_loop_invariant(i <= n && k == 0)
  {
    i++;
  }
  __CPROVER_assert(k == 0, "k == 0");
  return 0;
}
