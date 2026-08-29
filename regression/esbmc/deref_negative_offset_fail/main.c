/* A write one element below an object's base must be reported. The check was
   written as offset + access_sz > data_sz, which wraps back into range for
   every offset in [-access_sz, 0), so exactly the p[-1] shape went unreported. */
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
