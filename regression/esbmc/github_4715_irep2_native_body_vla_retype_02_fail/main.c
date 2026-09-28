// The branch/loop half of github_4715_irep2_native_body_vla_retype_01_fail: the
// native if/while/do-while/for arms fold the condition into a guard verbatim,
// so they need the same symbol-table retype the statement arms do. Review found
// this after the statement arms were fixed; the sweep sample contains no VLA in
// a branch condition. ASan: dynamic-stack-buffer-overflow.
int main(void)
{
  int n = 1;
  int a[n];
  a[0] = 42;
  n = 100;
  if (a[5] == 42)
    return 1;
  return 0;
}
