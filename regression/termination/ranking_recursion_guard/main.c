/* Soundness guard: the ranking checker must NOT certify a program whose
 * non-termination comes from RECURSION, not loops. This program has no
 * loops at all — rec() recurses unboundedly (the guard x<=23 && x>=-42
 * stays satisfiable forever because y feeds back into x). A loop-only
 * ranking analysis sees zero loops and would vacuously declare
 * termination — an unsound wrong-true. has_recursion() detects the
 * call-graph cycle (rec -> rec) and forces UNKNOWN.
 *
 * Expected verdict: VERIFICATION UNKNOWN (never SUCCESSFUL). */

extern int __VERIFIER_nondet_int(void);

void rec(int x, int y)
{
  if (x <= 23 && x >= -42)
    rec(2 * y - 2, x + 1);
}

int main()
{
  int n = __VERIFIER_nondet_int();
  rec(n, n + 1);
  return 0;
}
