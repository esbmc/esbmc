/* Regression test: stdin/stdout/stderr must not trigger "extern variable not
 * found" warnings.  They are defined in the ESBMC libc model (stdio.c). */
#include <stdio.h>
#include <assert.h>

int main()
{
  /* Verify the file pointers are non-null (defined, not nil). */
  assert(stdin != (void *)0);
  assert(stdout != (void *)0);
  assert(stderr != (void *)0);
  return 0;
}
