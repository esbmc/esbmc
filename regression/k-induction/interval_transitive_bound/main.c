/* Post-k-induction interval bound for a TRANSITIVELY-modified loop
 * variable.
 *
 * `state` is a global that main's loop modifies only indirectly, via
 * the call to step(). It never appears textually in main's loop body
 * (the body just reads `in` and calls step(in)). The interval domain
 * proves state stays in [0,2] (step assigns it only the constants
 * 0/1/2 under guards).
 *
 * k-induction's make_nondet_assign havocs state (it's in the loop's
 * transitively-modified set), emitting `state = NONDET()` in the havoc
 * block before the loop head. instrument_loop_bounds_after_kind must
 * then pin state to its interval at the inductive-step loop head.
 * Before the fix, that pass collected only symbols appearing textually
 * in the loop body, so the callee-modified `state` was missed and the
 * inductive step ran from an arbitrary state value. The fix also
 * collects the targets of the havoc block, so state gets bounded.
 *
 * The test inspects the transformed GOTO program: an
 * ASSUME(0 <= state <= 2) must be present (the post-havoc bound on the
 * transitively-modified variable). */

extern int __VERIFIER_nondet_int(void);

int state = 0;

void step(int in)
{
  if (state == 0)
  {
    if (in)
      state = 1;
  }
  else if (state == 1)
  {
    if (in)
      state = 2;
    else
      state = 0;
  }
  else
  {
    state = 0;
  }
}

int main()
{
  while (1)
  {
    int in = __VERIFIER_nondet_int();
    step(in);
  }
  return 0;
}
