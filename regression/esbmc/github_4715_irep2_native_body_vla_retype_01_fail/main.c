// A VLA's bound is snapshotted into a temporary at its declaration (C11
// 6.7.6.2p5), and convert_decl retypes the array symbol to that temporary
// mid-body. migrate_expr re-reads a level0 symbol's type from the symbol table,
// so the legacy path picks the retype up; a native arm storing the frontend's
// code2 verbatim kept `int[n]` and let the reassignment below widen the bound
// check into a vacuous one, silently accepting this overflow (W1-loc,
// esbmc/esbmc#4715). ASan calls it a dynamic-stack-buffer-overflow.
int main(void)
{
  int n = 1;
  int a[n];
  a[0] = 42;
  n = 100;
  return a[5];
}
