// Each call puts its operand of && on its own line, and the whole condition
// is a return value, not a branching statement: the return gets no branching
// waypoint. The if that tests the result keeps its own.
extern int __VERIFIER_nondet_int(void);
extern void reach_error(void);
int a, b;
int one(void) { return 1; }
int pa(void) { return a > 0; }
int pb(void) { return b > 0; }
int inv(void)
{
  return (
    one() &&
    pa() &&
    one() &&
    pb());
}
int main(void)
{
  a = __VERIFIER_nondet_int();
  b = __VERIFIER_nondet_int();
  if (!inv())
    return 0;
  if (a == 5)
    reach_error();
  return 0;
}
