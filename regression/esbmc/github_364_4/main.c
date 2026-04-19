#include <stdlib.h>

union a
{
};
struct b
{
  struct c *d;
} typedef *e;
struct f
{
  union a g;
};
struct b j, k;
struct c
{
  struct f h;
  void *i;
}

l(e m)
{
  m->d++;
  m->d = NULL;
}

void n(e m)
{
  m->d->h.g;
}

int main()
{
  int o;
  struct c p[1];
  k.d = p;
  j = k;
  if(o)
    l(&j);
  n(&j);
}

