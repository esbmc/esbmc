/* #11006: goto-cc stores __CPROVER_loop_invariant on the loop's back edge. */
unsigned nondet_uint(void);
int main()
{
  unsigned exp = nondet_uint(), e = exp, base = 3, acc = 1;
  while (e > 1)
    __CPROVER_loop_invariant(e <= exp)
  {
    if (e & 1)
      acc *= base;
    base *= base;
    e /= 2;
  }
  __CPROVER_assert(e <= exp, "e never grows");
  return 0;
}
