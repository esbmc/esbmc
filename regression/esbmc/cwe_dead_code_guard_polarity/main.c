extern int __VERIFIER_nondet_int(void);

// Pins the polarity of the guard named in a dead-code advisory: a probe
// assert(c) that is never violated proves !c infeasible, so the advisory names
// the opposite direction. `flag` also pins that the negation is derived from
// the expression rather than by flipping the probe's comment text, which would
// print "!flag" here — from_expr parenthesises by precedence.
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
