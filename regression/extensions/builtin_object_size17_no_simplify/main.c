#include <assert.h>

struct S
{
  int a;
  int b;
};

struct S arr[4];

int main()
{
  struct S *p = arr;

  // The whole object is the array, not the element the pointer's subtype
  // names: GCC reports 32 here, not sizeof(struct S).
  assert(__builtin_object_size(p, 0) == 32);
  assert(__builtin_object_size(p + 1, 1) == 24);

  return 0;
}
