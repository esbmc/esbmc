// KNOWNBUG: spanning-set coverage percentage is inflated because the
// numerator (reached_claims.size()) is NOT filtered against
// k_path_spanning_redundant before being divided by the maximal-only
// spanning_size_ denominator. See bmc.cpp:929 and bmc.cpp:947 — and
// the JSON output at bmc.cpp:1003-1007 already uses the redundant set
// to mark each claim's feasibility, so the data is available; the
// percentage formula just forgot to use it.
//
// This program has two correlated branches: `taken := (a > 0)` makes
// the second branch fully determined by the first.
//
// k-path emission at N=2 produces 6 asserts (2 length-1 at b1 plus
// 4 length-2 at b2). Spanning set keeps the 4 length-2 maximal goals;
// the 2 length-1 at b1 are subsumed (redundant).
//
// Reachability:
//   - both length-1 at b1: ALWAYS reached (any execution takes one
//     polarity at the first branch).
//   - length-2 {(a>0,T),(taken,T)}: reachable (a>0 -> taken=1).
//   - length-2 {(a>0,T),(taken,F)}: UNREACHABLE (a>0 -> taken=1).
//   - length-2 {(a>0,F),(taken,T)}: UNREACHABLE (a<=0 -> taken=0).
//   - length-2 {(a>0,F),(taken,F)}: reachable.
//
// So reached_claims contains 4 entries: 2 subsumed + 2 maximal.
// spanning_size_ = 4.
//
// Correct spanning-set coverage (Marré & Bertolino, IEEE TSE 2003):
//   |reached ∩ maximal| / |maximal|  =  2 / 4  =  50%
//
// Current (buggy) output:
//   |reached_all|       / |maximal|  =  4 / 4  =  100%
//
// The bug is asymmetric: numerator uses all reached, denominator
// uses maximal-only. bmc.cpp:943-944 even acknowledges the
// asymmetry ("Cap at 100% in case tracked_instance exceeds
// |spanning_set| due to subsumed-and-reached emissions") and works
// around it with a clamp instead of fixing the formula.
//
// Fix sketch: at bmc.cpp:929, replace
//     const size_t tracked_instance = reached_claims.size();
// with a loop that excludes claim signatures whose (msg, loc) is in
// goto_coveraget::k_path_spanning_redundant. Then update test 4
// (50%) and test 12 (25%) expectations accordingly, and revisit the
// Reached line and JSON percentage for the same asymmetry.
//
// When fixed: promote KNOWNBUG -> CORE; remove this comment block.
int main()
{
  int a;
  int taken;
  if (a > 0)
    taken = 1;
  else
    taken = 0;

  if (taken)
    a = a;
  else
    a = -a;

  return a;
}
