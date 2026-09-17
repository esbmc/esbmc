/* Regression for #7434 review: with more than one input file,
 * options["input-file"] retains only the last positional argument, so every
 * step from this file looked foreign to input_file_check. The target waypoint
 * was hoisted past it to main in second.c and the function_return waypoint was
 * dropped outright. Both belong here, on line 11 and line 12. */
#include <assert.h>
int __VERIFIER_nondet_int(void);

int check(void)
{
  int v = __VERIFIER_nondet_int();
  assert(v != 7);
  return v;
}
