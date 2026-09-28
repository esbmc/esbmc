extern int __VERIFIER_nondet_int(void);

// No dead code, but --sarif-output is requested: consumers expect a valid
// SARIF run with an empty results array, not a missing document.
int main(void)
{
  int x = __VERIFIER_nondet_int();
  if (x > 5)
    return 0;
  else
    return 1;
}
