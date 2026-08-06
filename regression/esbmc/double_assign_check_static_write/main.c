// Discriminating sibling of double_assign_check_expired_write, one keyword
// apart: a static has whole-program lifetime, so no frame teardown touches its
// L2 record and the write through the returned pointer is well defined.
#include <assert.h>

static int *not_dangling(void)
{
  static int cell = 42;
  return &cell;
}

int main(void)
{
  int *p = not_dangling();
  *p = 7;
  assert(*p == 7);
  return 0;
}
