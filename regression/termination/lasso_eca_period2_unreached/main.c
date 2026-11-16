// As lasso_eca_period2, but main returns before the loop, so the cycle the
// inputs could follow is never entered.

int a = 0;
int skip = 1;

extern int __VERIFIER_nondet_int(void);
extern void exit(int);

int update(int input)
{
  if (a == 0 && input == 1)
  {
    a = 1;
    return 1;
  }
  if (a == 1 && input == 2)
  {
    a = 0;
    return 2;
  }
  if (a == 0 || a == 1)
    exit(0);
  if (a != 0)
    exit(0);
  return -2;
}

int main(void)
{
  int sink = 0;
  if (skip)
    return 0;
  while (1)
  {
    int input;
    input = __VERIFIER_nondet_int();
    if (input != 1 && input != 2)
      return -2;
    sink = update(input);
  }
  return sink;
}
