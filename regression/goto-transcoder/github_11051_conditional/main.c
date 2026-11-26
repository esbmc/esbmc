/* A call past a short-circuit operand is not always evaluated, so it cannot
 * be hoisted ahead of the invariant; the invariant is dropped. */
unsigned nondet_uint(void);
_Bool closure(unsigned *p)
{
  return *p <= 4;
}
int main()
{
  unsigned *p = 0;
  for (unsigned i = 0; i < 4; i++)
    __CPROVER_loop_invariant(p == 0 || closure(p))
  {
  }
  return 0;
}
