/* A CBMC goto-binary whose symbol table names a __KANI_ helper, as every Kani
 * binary does, so ESBMC verifies it by k-induction by default. */
_Bool nondet_bool(void);
int __KANI_marker;
int main(void)
{
  unsigned x = 0;
  while (nondet_bool())
  {
    x = (x + 1) % 4;
    __CPROVER_assert(x < 4, "x stays below 4");
  }
  return 0;
}
