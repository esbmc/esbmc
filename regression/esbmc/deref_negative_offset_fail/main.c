/* R35: a write below an object's base is neither performed nor flagged.
   The symmetric overflow, *(int *)(base + 16), is caught. */
#include <assert.h>

struct c
{
  int f0, f1, f2, f3;
};

int main(void)
{
  struct c s = {0, 0, 0, 0};
  char *base = (char *)&s;

  *(int *)(base - 4) = 42;

  assert(s.f0 == 0 && s.f1 == 0 && s.f2 == 0 && s.f3 == 0);
  return 0;
}
