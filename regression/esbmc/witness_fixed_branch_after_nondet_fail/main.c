// No nondet visit of line 8 follows f(0), so no later waypoint can bind to
// it and the fixed visit gets none (#9689).
extern int __VERIFIER_nondet_int(void);
extern void reach_error(void);

int f(int m)
{
  if (m < 10)
    return 0;
  return 1;
}

int main(void)
{
  int x = __VERIFIER_nondet_int();
  int b = f(x);
  int a = f(0);
  if (a + b != 0)
    reach_error();
  return 0;
}
