#include <assert.h>

struct c
{
  int f0, f1, f2, f3, f4, f5;
};

int main(void)
{
  struct c s = {0, 0, 0, 0, 0, 0};

  void *member = &s.f2;
  *(int *)(member - 8) = 42;

  assert(s.f0 == 0);
  return 0;
}
