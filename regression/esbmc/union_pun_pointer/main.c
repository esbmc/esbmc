#include <assert.h>

struct pair
{
  int *first;
};

union scalar_or_pair
{
  struct pair p;
  int *raw;
};

struct left
{
  int *lp;
};

struct right
{
  int *rp;
};

union arms
{
  struct left l;
  struct right r;
};

int main()
{
  int x = -2;

  union scalar_or_pair u;
  u.p.first = &x;
  assert(*u.raw == -2);

  u.raw = &x;
  assert(*u.p.first == -2);

  union arms a;
  a.l.lp = &x;
  assert(*a.r.rp == -2);

  *u.raw = 7;
  assert(x == 7);

  return 0;
}
