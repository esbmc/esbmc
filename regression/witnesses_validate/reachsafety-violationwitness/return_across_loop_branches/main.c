extern int __VERIFIER_nondet_int(void);
extern void reach_error(void);
int main(void)
{
  int state = __VERIFIER_nondet_int();
  while (state < 2)
  {
    if (state == 0)
    {
      int a = __VERIFIER_nondet_int();
      if (a != 1)
        return 0;
      state = 1;
    }
    else
    {
      int b = __VERIFIER_nondet_int();
      if (b != 0)
        return 0;
      state = 2;
    }
  }
  reach_error();
}
