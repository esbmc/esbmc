/* Library-loop marker pattern. main() has no loop of its own; the
 * only loop in the program lives in __memset_impl, a body.hide
 * operational model linked in for the memset() call.
 *
 * Two passes in goto_termination treat body.hide functions
 * differently:
 *
 *   - The IS-reliability gate (loop_is_is_unreliable) and no-op-cycle
 *     injection SKIP body.hide helpers. Otherwise a helper like
 *     __ESBMC_atexit_handler — linked into every program — would
 *     classify as the empty-modified-set-with-assign hazard and flip
 *     the program-wide gate, disabling IS for every benchmark.
 *
 *   - The marker pass RUNS on body.hide helpers. Without a marker in
 *     __memset_impl, this program would carry zero termination claims;
 *     IS would return a vacuous UNSAT and the BMC layer would report a
 *     spurious non-termination witness (wrong-false).
 *
 * With the marker present, FC unwinds memset's 80-iteration loop and
 * proves termination. The default --max-k-step 50 stops short of 80,
 * so we raise it to 100; FC then closes at k = 81. Expected verdict:
 * VERIFICATION SUCCESSFUL. */

typedef unsigned int size_t;
extern void *memset(void *s, int c, size_t n);

int main()
{
  char buf[80];
  memset(buf, 0, 80);
  return 0;
}
