/* Soundness/robustness guard for the invariant synthesizer's interaction
 * with body assignments from NONDET.
 *
 * The loop `while (i < n) { v = nondet(); s = s+v; i = i+1; }` is
 * terminating by the bare ranking argument m = n - i: bounded and
 * strictly decreases by 1 each iteration. The invariant synthesizer also
 * runs (constant seeds v=s=i=0 from the pre-header), but every candidate
 * atom on `v` has a post-state of NONDET, an opaque sideeffect we cannot
 * faithfully widen into the int64 atom domain. widen_arith returns nil
 * for such expressions and the atom-builder skips them, so the synthesis
 * produces no atoms for the affected variables (it does for s/i, which
 * are deterministic) and the obligations discharge unchanged.
 *
 * A regression where widen_arith fell back to a generic typecast of the
 * sideeffect would hand the solver an expression it cannot reason about,
 * stalling or timing out on what should be an immediate certificate.
 *
 * Expected verdict: VERIFICATION SUCCESSFUL. */

extern unsigned char __VERIFIER_nondet_uchar(void);

int main()
{
  unsigned char n = __VERIFIER_nondet_uchar();
  if (n == 0)
    return 0;
  unsigned char v = 0;
  unsigned int s = 0;
  unsigned int i = 0;
  while (i < n)
  {
    v = __VERIFIER_nondet_uchar();
    s += v;
    ++i;
  }
  return 0;
}
