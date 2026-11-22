#include <assert.h>

struct pair
{
  int q;
  int r;
};

struct pair f(int n)
{
  struct pair p;
  p.q = n / 2;
  p.r = n % 2;
  return p;
}

int main(void)
{
  struct pair d = f(7);
  assert(d.q == 3);
  assert(d.r == 1);
  return 0;
}
