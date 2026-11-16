// f(0) fixes the branch at line 9, f(x) does not. A validator binds the
// waypoint of f(x)'s visit to the next visit of line 9, f(0)'s, unless that
// visit has a waypoint of its own.
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
  int a = f(0);
  int b = f(x);
  if (a + b != 0)
    reach_error();
  return 0;
}
