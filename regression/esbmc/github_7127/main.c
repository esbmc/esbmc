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

  void *bytes = (char *)&s + 8;
  *(int *)(bytes - 4) = 7;

  *(int *)(member + 8) = 5;
  *(int *)(member - (-4)) = 6;

  assert(s.f0 == 42);
  assert(s.f1 == 7);
  assert(s.f2 == 0);
  assert(s.f3 == 6);
  assert(s.f4 == 5);
  return 0;
}
