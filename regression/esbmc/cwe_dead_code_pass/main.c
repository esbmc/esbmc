extern int __VERIFIER_nondet_int(void);

int main(void)
{
  int x = __VERIFIER_nondet_int();
  // Both branches are reachable, so there is no dead code to report.
  if (x > 5)
    return 0;
  else
    return 1;
}
