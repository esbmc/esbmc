// Issue #9887: the cycle waypoint must lie on a loop that can run forever.
extern int __VERIFIER_nondet_int(void);
int drain(void)
{
  int x = 1;
  while (x)
    x = __VERIFIER_nondet_int();
  return x;
}
int main(void)
{
  int x = __VERIFIER_nondet_int();
  while (x > 0)
    x = 1;
  return 0;
}
