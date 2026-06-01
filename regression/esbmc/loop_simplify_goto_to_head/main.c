/* Soundness: refuse Path 1 / Path 2 rewrites when loop_head has any
 * incoming GOTO other than the back-edge. An external `goto L; … L:
 * while(...)` would otherwise land on the synthesized head (assume or
 * ASSIGN i=post), seeing its effect, and bypass the pre-loop init
 * that was supposed to determine the rewritten value.
 *
 * Fallthrough: i=0 → loop steps 0,3,6,9,12 → exits at i=12.
 * Goto path:   i=1 → loop steps 1,4,7,10  → exits at i=10.
 * Path 2 would commit to init=0 (parse_init's textually nearest
 * ASSIGN) and rewrite the head to ASSIGN i=12, masking the i=10 path
 * and the assertion failure it produces. */
#include <assert.h>
int nondet_int();
int main()
{
  int i = 1;
  if (nondet_int())
    goto L;
  i = 0;
L:
  while (i < 10)
    i += 3;
  __ESBMC_assert(i == 12, "goto bypasses pre-loop init");
  return 0;
}
