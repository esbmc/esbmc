/* Self-recursion termination by ranking over a parameter.
 *
 * rec(x) recurses under path condition x > 0 with argument x-1. The
 * measure m = x (from x > 0) strictly decreases across the recursive
 * call (x-1 < x) and is bounded below (x > 0 => x >= 1), so the
 * recursion is well-founded. prove_self_recursion_terminates extracts
 * the call's path condition and argument, builds m and m(args) = m with
 * the formal x replaced by x-1, and discharges:
 *   bounded:  x>0 AND (m<1)        UNSAT
 *   decrease: x>0 AND (m(x-1)>=m)  UNSAT
 *
 * Expected verdict: VERIFICATION SUCCESSFUL. */
extern int __VERIFIER_nondet_int(void);
void rec(int x)
{
  if (x > 0)
    rec(x - 1);
}
int main()
{
  rec(__VERIFIER_nondet_int());
  return 0;
}
