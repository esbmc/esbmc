/* Constant-bound counter loop: termination follows from the loop
 * being a recognized counter pattern with literal init/step/bound.
 *
 * FC at k=2 (the default for --termination) can't unwind 1000000
 * iterations, so without step recognition the verdict would be
 * UNKNOWN. goto_loop_simplify's Path 2 (try_step_recognition) is now
 * enabled under --termination: it parses `for(i=0; i<1000000; i++)`,
 * computes the post-value 1000000 in BigInt (verifying it fits the
 * induction variable's bit-width), and rewrites the loop to a single
 * ASSIGN i = 1000000 followed by SKIPs. FC then closes at k=1.
 *
 * parse_init walks back from loop_head looking for the ASSIGN i = 0
 * pre-loop init. After goto_termination's havoc, the immediate
 * predecessor of loop_head is the inductive-step NONDET ASSIGN and
 * an ASSUME, not the original init. parse_init now skips
 * inductive_step_instruction entries, so the original init is still
 * found and step recognition fires. */

int main()
{
  int i;
  for (i = 0; i < 1000000; i++)
  {
  }
  return 0;
}
