int nondet_int(void);
int main(void)
{
  int d = nondet_int();
  int *p = nondet_int() ? 0 : &d;
  __CPROVER_assert(d >= 0 ==> 10 / d != 99, "div");
  __CPROVER_assert(p != 0 ==> *p == d, "ptr");
  return 0;
}
