#include <assert.h>
#include <stddef.h>

struct S
{
  int a;
  char b[8];
  int c;
};

enum
{
  v = offsetof(struct S, c)
};

static const size_t offsets[] = {offsetof(struct S, a), offsetof(struct S, c)};

int main(void)
{
  assert(offsets[1] == v);
  assert(v == 8);
  return 0;
}
