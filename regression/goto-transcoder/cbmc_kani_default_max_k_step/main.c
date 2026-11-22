/* A CBMC goto-binary whose symbol table names a __KANI_ helper, as every Kani
 * binary does, so ESBMC verifies it by k-induction by default. */
int __KANI_marker;
int main(void)
{
  unsigned x = 0;
  for (unsigned i = 0; i < 100; ++i)
  {
    ++x;
    __CPROVER_assert(x != 80, "x never reaches 80");
  }
  return 0;
}
