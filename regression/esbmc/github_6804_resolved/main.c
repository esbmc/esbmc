/* Companion to github_6804. An additive round trip keeps the value set
 * exhaustive, so s is recovered and the write must land -- the unresolved-target
 * failure must not fire here. */
#include <stdint.h>
#include <assert.h>

int main(void)
{
  struct S
  {
    int x;
  } s = {.x = 42};

  uintptr_t u = (uintptr_t)&s;
  u += 8;
  u -= 8;

  int *p = (int *)u;
  *p = 3;

  assert(s.x == 3);
  return 0;
}
