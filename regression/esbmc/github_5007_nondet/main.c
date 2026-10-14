extern int __VERIFIER_nondet_int(void);

// Non-singleton VLA size: the overflow2t guard's operand must NOT be collapsed
// to a constant, exercising the leaf-only optimisation path of the fix.
int main()
{
  int n = __VERIFIER_nondet_int();
  __ESBMC_assume(n > 0 && n < 100);
  long packet[n];
  return 0;
}
