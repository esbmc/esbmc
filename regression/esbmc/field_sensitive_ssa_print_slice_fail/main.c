#include <assert.h>

struct S
{
  int a;
  void *p;
};

int main(void)
{
  struct S s;
  assert(s.a == 0);
  return 0;
}
