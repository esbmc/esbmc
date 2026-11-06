// --goto-unwind copies the loop body, so each iteration's `if` is its own
// instruction. The first iteration's branch is fixed (c is 0) and still needs
// a "false" waypoint, or a validator binds the second iteration's "true" to it.
extern int __VERIFIER_nondet_int(void);
extern void reach_error(void);

int main(void)
{
  int x = __VERIFIER_nondet_int();
  for (int i = 0; i < 2; i++)
  {
    int c = i ? x : 0;
    if (c == 3)
      reach_error();
  }
  return 0;
}
