/* Mutual recursion is out of scope for the per-function ranking prover
 * (a measure must decrease around the whole call cycle, not within one
 * function). analyze_recursion classifies f<->g as `unsupported` and the
 * ranking check returns UNKNOWN, so this non-terminating program is NOT
 * wrongly certified as terminating.
 *
 * Expected verdict: VERIFICATION UNKNOWN (never SUCCESSFUL). */
extern int __VERIFIER_nondet_int(void);
void g(int x);
void f(int x) { if (x > 0) g(x + 1); }
void g(int x) { if (x > 0) f(x + 1); }
int main()
{
  f(__VERIFIER_nondet_int());
  return 0;
}
