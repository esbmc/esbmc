/* A CBMC goto-binary whose symbol table names a __KANI_ helper, as every Kani
 * binary does, so ESBMC verifies it by k-induction by default. */
int __KANI_marker;
int main(void)
{
  unsigned i = 0, x = 0;
  while (i < 5)
  {
    ++i;
    ++x;
    __CPROVER_assert(x <= 5, "x stays at most 5");
  }
  return 0;
}
