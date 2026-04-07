/* Regression test: stdin/stdout/stderr must not trigger "extern variable not
 * found" warnings.  They are silently nondet-initialized as known IO globals
 * (GitHub #3973). */
#include <stdio.h>

int main()
{
  return 0;
}
