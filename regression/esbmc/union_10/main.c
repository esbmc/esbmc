#include <stdlib.h>
union a
{
};
struct b
{
  struct c *d
} typedef *e;
struct f
{
  union a g
};
struct b j, k;
struct c
{
  struct f h;
  void *i
} l(e m)
{
  m->d++;
}
n(e m)
{
  m->d->h.g;
}
struct c p[];
main()
{
  int o;
  k.d = p;
  j = k;
  if(o)
    l(&j);
  n(&j);
}
