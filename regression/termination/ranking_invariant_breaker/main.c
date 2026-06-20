/* Negative test for the invariant synthesizer's base-case soundness.
 *
 * If d1 were unconditionally seeded as 73 from a syntactic backward scan,
 * the invariant d1 >= 1 would seem to hold at loop entry and the
 * decrease obligation would discharge — but along the else branch d1 = 0,
 * the loop genuinely does not terminate (x = x - 0 = x forever). To stay
 * sound, collect_seeds bails as soon as the prefix between function entry
 * and the loop head contains a GOTO/IF, a jump target, a function call,
 * or any memory-touching assign. The if/else here introduces an IF in the
 * prefix, so no seeds are collected, the invariant is `true`, and the
 * decrease obligation fails — the checker declines and the program is
 * left UNKNOWN by the existing forward-condition / inductive-step
 * machinery. A regression that relaxed the prefix check could spuriously
 * certify this non-terminating loop as VERIFICATION SUCCESSFUL.
 *
 * Expected verdict: VERIFICATION UNKNOWN. */

extern int __VERIFIER_nondet_int(void);

int main()
{
  int x = __VERIFIER_nondet_int();
  int d1;
  int d2 = 74;
  if (__VERIFIER_nondet_int())
    d1 = 73;
  else
    d1 = 0; /* on this path, x = x - 0 never makes progress */
  while (x >= 0)
  {
    x = x - d1;
    d1 = d2 + 1;
    d2 = d1 + 1;
  }
  return 0;
}
