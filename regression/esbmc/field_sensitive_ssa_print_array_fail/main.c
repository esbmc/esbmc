#include <assert.h>
#include <stdlib.h>

struct S
{
  int x;
  long pair[2];
};

int main(void)
{
  struct S t;
  t.x = 1;
  t.pair[0] = 6;
  t.pair[1] = 7;
  struct S *p = calloc(1, sizeof(struct S) + sizeof(long));
  if (!p)
    return 0;
  *p = t;
  struct S back = *p;
  assert(back.pair[1] != 7);
  return 0;
}
