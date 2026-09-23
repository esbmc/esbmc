// #7907: the vector member is not packed straight after the int before it.
#include <assert.h>
#include <stddef.h>

typedef int v4i __attribute__((__vector_size__(16)));

struct S
{
  int x;
  v4i v;
};

int main(void)
{
  assert(offsetof(struct S, v) == 4);
  return 0;
}
