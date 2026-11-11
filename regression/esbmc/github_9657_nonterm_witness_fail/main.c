// Issue #9657: a non-terminating loop yields a YAML cycle waypoint.
extern int __VERIFIER_nondet_int(void);
int main(void)
{
  int x = __VERIFIER_nondet_int();
  while (x > 0)
    x = 1;
  return 0;
}
