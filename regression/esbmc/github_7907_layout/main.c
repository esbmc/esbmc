// #7907: a struct lays a vector member out at the vector's alignment, as clang
// does, so offsetof and sizeof agree with the object ESBMC models.
#include <assert.h>
#include <stddef.h>
#include <string.h>

typedef int v4i __attribute__((__vector_size__(16)));

struct S
{
  int x;
  v4i v;
};

int main(void)
{
  assert(offsetof(struct S, v) == 16 && sizeof(struct S) == 32);

  struct S a = {7, {1, 2, 3, 4}}, b;
  memcpy(&b, &a, sizeof a);
  assert(b.x == 7 && b.v[2] == 3);

  struct S arr[2] = {{1, {1, 1, 1, 1}}, {2, {5, 6, 7, 8}}};
  v4i *p = (v4i *)((char *)&arr[1] + offsetof(struct S, v));
  assert((*p)[3] == 8);
  return 0;
}
