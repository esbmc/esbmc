#include <assert.h>
#include <setjmp.h>

static jmp_buf env;

int main(void)
{
  // No longjmp reaches this frame, so the direct return is the only one the
  // modelled control flow can take, and it is 0.
  int r = setjmp(env);
  assert(r == 0);

  int s = setjmp(env);
  assert(!s);

  return 0;
}
