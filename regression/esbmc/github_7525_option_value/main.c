// A value-taking option swallows the next --flag as its value (#7525).
int main(void)
{
  int x = __VERIFIER_nondet_int();
  if (x > 5)
    goto ERROR;
  return 0;
ERROR:
  return 1;
}
