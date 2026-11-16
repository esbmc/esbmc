// Issue #9886: drain's loop sorts first but nothing calls drain, so the cycle
// waypoint names main's loop, which main reaches past a call that returns.
extern int __VERIFIER_nondet_int(void);
int drain(void)
{
  int x = 1;
  while (x)
    x = __VERIFIER_nondet_int();
  return x;
}
int seed(void)
{
  return __VERIFIER_nondet_int();
}
int main(void)
{
  int x = seed();
  while (x > 0)
    x = 1;
  return 0;
}
