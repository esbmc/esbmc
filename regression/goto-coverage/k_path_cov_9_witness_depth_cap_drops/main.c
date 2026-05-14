// Regression for #4330 review (witness-depth cap path). Three sequential
// branches on distinct variables produce 14 witnesses at the default
// depth cap (--k-path-witness-depth=8): 2 (branch 1) + 4 (branch 2) +
// 8 (branch 3). With a tight cap of 2, post-simplification expression
// depth strictly above 2 is dropped at instrumentation time. Branch 1's
// single-atom witnesses (e.g. `a > 0`, depth 2) survive; branch 2/3
// chained-conjunction witnesses (depth >= 3) are dropped.
//
// Soundness rationale: dropping a witness removes it from BOTH the
// numerator (reached_claims for that goal) and the denominator
// (total_kpath / all_claims). The reported percentage stays accurate
// over the emitted subset; the user simply gets a less complete
// measurement, which is the documented contract of the tractability
// knob. The test asserts that:
//   - the reduced witness count is observed (2, not 14),
//   - the surviving subset is fully reached (100%, not a vacuous N/A
//     and not a false 0%).
int main()
{
  int a, b, c;
  if (a > 0)
    ;
  else
    ;
  if (b > 0)
    ;
  else
    ;
  if (c > 0)
    ;
  else
    ;
}
