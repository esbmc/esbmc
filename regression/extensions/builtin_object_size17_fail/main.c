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

  // sizeof(*p), the answer sizing by the pointer's subtype gives.
  assert(__builtin_object_size(p, 0) == 8);

  return 0;
}
