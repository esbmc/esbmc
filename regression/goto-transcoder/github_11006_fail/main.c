unsigned nondet_uint(void);
int main()
{
  unsigned n = nondet_uint(), i = 0;
  __CPROVER_assume(n <= 5);
  while (i < n)
    __CPROVER_loop_invariant(i < n)
  {
    i++;
  }
  __CPROVER_assert(i == n, "loop stops at n");
  return 0;
}
