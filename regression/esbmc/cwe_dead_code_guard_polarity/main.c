extern int __VERIFIER_nondet_int(void);

// Pins both polarities of the guard named in a dead-code advisory. `flag` also
// pins expression-level negation: the surviving probe's comment is "!flag", so
// printing it gives "!flag" and wrapping it in !(...) gives "!(!flag)".
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
