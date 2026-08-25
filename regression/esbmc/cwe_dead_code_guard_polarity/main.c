extern int __VERIFIER_nondet_int(void);

// Pins both polarities of the guard named in a dead-code advisory. `flag` also
// pins that the negation is derived from the expression: flipping the probe's
// comment text would print "!flag" here.
int main(void)
{
  _Bool flag = 0;
  int x = __VERIFIER_nondet_int();
  if (flag)
    return 1;
  __ESBMC_assume(x > 10);
  if (x > 5)
    return 0;
  return 2;
}
