// Issue #9657: a terminating loop yields no non-termination witness.
extern int __VERIFIER_nondet_int(void);
int main(void)
{
  int x = __VERIFIER_nondet_int();
  while (x > 0)
    x--;
  return 0;
}
