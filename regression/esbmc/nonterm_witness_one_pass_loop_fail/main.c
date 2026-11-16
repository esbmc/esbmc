// Issue #9886: advance's loop sorts first, but every pass from its head leaves
// it (a CIL-style coroutine), so the cycle waypoint names main's loop.
extern int __VERIFIER_nondet_int(void);
int pc, n;
void advance(void)
{
  if (pc == 1)
    goto resume;
  while (n < __VERIFIER_nondet_int())
  {
    n = n + 1;
    pc = 1;
    goto out;
  resume:;
  }
out:;
}
int main(void)
{
  int x = __VERIFIER_nondet_int();
  while (x > 0)
  {
    if (x == 0)
      advance();
    x = 1;
  }
  return 0;
}
