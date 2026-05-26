/* Vacuous-UNSAT inductive-step verdict. main() has no loops of its
 * own; the only loop in the program lives in __memset_impl (a
 * body.hide library helper). The marker pass skips body.hide
 * functions, so symex emits zero termination claims. The SMT solver
 * trivially returns UNSAT on an empty claim set, and without the
 * vacuous-UNSAT guard in is_inductive_step_violated the BMC layer
 * would interpret that as a non-termination witness — yielding
 * VERIFICATION FAILED (wrong-false).
 *
 * The fix in bmc.cpp flags `bmc-vacuous-unsat` when remaining_claims
 * hits zero; is_inductive_step_violated consults the flag and
 * returns TV_UNKNOWN rather than TV_FALSE in that case. FC is then
 * free to keep unwinding past memset's 80-iteration bound; with
 * --max-k-step 100 it succeeds at k = 81. Expected verdict:
 * VERIFICATION SUCCESSFUL. */

typedef unsigned int size_t;
extern void *memset(void *s, int c, size_t n);

int main()
{
  char buf[80];
  memset(buf, 0, 80);
  return 0;
}
