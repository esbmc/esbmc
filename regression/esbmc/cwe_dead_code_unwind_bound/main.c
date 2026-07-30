// A loop needing 100 iterations run under --unwind 4: the iterations beyond
// the bound are cut, so a branch that is live only past the bound reads as
// unreachable. This is inherent to bounded model checking, not true dead code,
// so the advisory is explicitly scoped "up to the current unwinding bound".
// Documents the interaction of --dead-code-check with loops and --unwind.
int main(void)
{
  int a[100];
  int i;
  for (i = 0; i < 100; i++)
    a[i] = i;
  return a[0];
}
