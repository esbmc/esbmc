void reach_error(void) {}
int __VERIFIER_nondet_int(void);

int main(void)
{
  int x = __VERIFIER_nondet_int();
  int flag = 0;
  if (x < 0)
    flag = 1;
  if (flag)
    reach_error();
  return 0;
}
