/* A CBMC goto-binary whose symbol table names a __KANI_ helper, as every Kani
 * binary does. The loop has no bound, so only k-induction proves it. */
_Bool nondet_bool(void);
int __KANI_marker;
int main(void)
{
  unsigned x = 0;
  while (nondet_bool())
  {
    x = (x + 1) % 4;
    __CPROVER_assert(x < 3, "x stays below 3");
  }
  return 0;
}
