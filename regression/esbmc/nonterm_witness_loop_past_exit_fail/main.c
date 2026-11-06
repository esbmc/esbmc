// Issue #9886: cleanup's loop sorts first, but main calls cleanup only past
// its own loop's exit, which the inductive step shows unreachable, so the
// cycle waypoint names main's loop.
extern int __VERIFIER_nondet_int(void);
int sum;
void cleanup(void)
{
  int i = 0;
  while (1)
  {
    if (i >= 3)
      break;
    sum = sum + __VERIFIER_nondet_int();
    i = i + 1;
  }
}
int main(void)
{
  int x = __VERIFIER_nondet_int();
  while (x > 0)
    x = 1;
  cleanup();
  return 0;
}
