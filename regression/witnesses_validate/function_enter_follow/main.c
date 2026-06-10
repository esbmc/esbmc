void reach_error(void) {}
int __VERIFIER_nondet_int(void);

static void check(int x)
{
  if (x < 0)
    reach_error();
}

int main(void)
{
  int a = __VERIFIER_nondet_int();
  int b = __VERIFIER_nondet_int();
  if (a > 0)
    check(a);
  if (b < 0)
    check(b);
  return 0;
}
