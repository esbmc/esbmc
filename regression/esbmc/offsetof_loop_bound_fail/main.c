#include <stddef.h>

struct P
{
  int a;
  int b;
};

int main()
{
  unsigned n = (unsigned)offsetof(struct P, b);
  int s = 0;
  for (unsigned i = 0; i < n; i++)
    s++;
  __ESBMC_assert(s == 9, "b does not sit nine bytes into the struct");
  return 0;
}
