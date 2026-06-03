void reach_error(void) {}
int __VERIFIER_nondet_int(void);

static int compute(int x)
{
  return x * 2;
}

int main(void)
{
  int a = __VERIFIER_nondet_int();
  int r = compute(a);
  if (r < 0)
    reach_error();
  return 0;
}
