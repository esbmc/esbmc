// GNU local labels. `__label__ l;` only scopes the name -- the label is placed
// by its statement -- but the declaration used to convert to an expression,
// and a DeclStmt's operands must be statements, so goto-convert aborted on
// "label". Reduced from gcc.c-torture/execute/930406-1.c and 980526-1.c, both
// of which this fixes (issue #4076).

// 980526-1.c's shape: a static jump table over two local labels, indexed at
// run time -- the interpreter idiom the extension exists for.
static int dispatch(int x)
{
  __label__ lbl1;
  __label__ lbl2;
  static void *jtab[2];
  jtab[0] = &&lbl1;
  jtab[1] = &&lbl2;
  goto *jtab[x];
lbl1:
  return 1;
lbl2:
  return 2;
}

int main(void)
{
  __ESBMC_assert(dispatch(0) == 1, "jump table selects the first label");
  __ESBMC_assert(dispatch(1) == 2, "jump table selects the second label");

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
