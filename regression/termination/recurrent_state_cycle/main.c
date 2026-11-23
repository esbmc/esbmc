// For 1 <= n <= 9 the loop goes back to i = 0 and runs forever; for any other
// n it ends. The inductive step cannot show this: from most states the loop
// exits.
int __VERIFIER_nondet_int(void);

int main(void)
{
  int i = 0, n = __VERIFIER_nondet_int();
  while (i < 10)
  {
    i++;
    if (i == n)
      i = 0;
  }
  return 0;
}
