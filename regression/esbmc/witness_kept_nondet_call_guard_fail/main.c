// y is odd, so the call at line 14 never runs. Its result is unused, but the
// witness must not name it: a validator would bind that function_return
// waypoint to the call at line 15 and take the else branch.
extern int __VERIFIER_nondet_int(void);
extern void reach_error(void);

int main(void)
{
  int y = __VERIFIER_nondet_int() | 1;
  int a = 0;
  if (y != 0)
    a = 1;
  else
    a = __VERIFIER_nondet_int();
  int b = __VERIFIER_nondet_int();
  if (b == 5)
    reach_error();
  return 0;
}
