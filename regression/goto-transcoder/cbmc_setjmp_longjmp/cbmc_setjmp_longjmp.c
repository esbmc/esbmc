#include <assert.h>
#include <setjmp.h>

static jmp_buf env;

int main(void)
{
  int r = setjmp(env);
  if (r == 0)
    longjmp(env, 3);
  // Neither CBMC nor ESBMC models the longjmp transfer, so control never comes
  // back through setjmp with 3 and this assertion fails in both. Pinned so the
  // setjmp rewrite cannot quietly grow into a claim that longjmp works.
  assert(r == 3);
  return 0;
}
