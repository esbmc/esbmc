/* Regression for #7434: the nondet assignment inside nondet.h is on line 29 of
 * that header, past the end of this file. See nondet.h for the full rationale. */
#include "nondet.h"
#include <assert.h>

int main(void)
{
  int a = from_header();
  int b = __VERIFIER_nondet_int();
  assert(a + b != 7);
  return 0;
}
