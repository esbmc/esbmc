// An if without an else whose && has calls becomes nested ifs, one per
// operand, so neither names the condition's direction. The later if keeps its
// waypoint.
extern int __VERIFIER_nondet_int(void);
extern void reach_error(void);
int a, b, c;
int pa(void) { return a > 0; }
int pb(void) { return b > 0; }
int main(void)
{
  a = __VERIFIER_nondet_int();
  b = __VERIFIER_nondet_int();
  c = __VERIFIER_nondet_int();
  if (c == 3)
    return 0;
  if (pa() &&
      pb())
    reach_error();
  return 0;
}
