// GNU local labels. `__label__ l;` only scopes the name -- the label is placed
// by its statement -- but the declaration used to convert to an expression,
// and a DeclStmt's operands must be statements, so goto-convert aborted on
// "label". Reduced from gcc.c-torture/execute/930406-1.c (issue #4076).
int main(void)
{
  int x = 1;

  ({
    __label__ mylabel;
  mylabel:
    x++;
    if (x != 3)
      goto mylabel;
  });

  __ESBMC_assert(x == 3, "the local-label loop runs to completion");

  // A local label also has an address, and jumping through it must agree.
  int hops = 0;
  {
    __label__ again;
    void *target = &&again;
  again:
    hops++;
    if (hops < 2)
      goto *target;
  }
  __ESBMC_assert(hops == 2, "computed goto to a local label");
  return 0;
}
