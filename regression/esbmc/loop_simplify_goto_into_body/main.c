/* Soundness: body_is_safe must reject loops whose body contains a
 * goto-target instruction. An external GOTO landing inside the body
 * would otherwise bypass the synthesized loop_head rewrite (an
 * ASSIGN for Path 2, an assume for Path 1) and fall through past the
 * SKIPs with the pre-loop value. */
#include <assert.h>
int nondet_int();
int main()
{
  int i = 99;
  int j = 99;
  if (nondet_int())
    goto L;
  i = 0;
  j = 0;
  while (i < 10)
  {
    j++;
    L:
    i++;
  }
  /* Goto path: i remains 99, jumps to L, runs i++ → i=100, !(100<10)
   * exits the loop with i=100. The assertion must fail. */
  __ESBMC_assert(i == 10, "external goto into body");
  return 0;
}
