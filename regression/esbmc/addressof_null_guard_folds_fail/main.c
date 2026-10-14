/* The address of a declared object is never null, so a null guard on it is
   decidable. Each shape below bounds a loop by a value guarded that way; if
   the guard does not fold, the bound stays symbolic and the loop unwinds to
   --unwind instead of its real trip count. */
#include <assert.h>

struct L
{
  int items[10];
  unsigned long size;
};

static unsigned long via_ternary(const struct L *l)
{
  return l ? l->size : 0;
}

static unsigned long via_neq(const struct L *l)
{
  return l != 0 ? l->size : 0;
}

static unsigned long via_early_return(const struct L *l)
{
  if (l == 0)
    return 0;
  return l->size;
}

static unsigned long total(const struct L *l, unsigned long n)
{
  unsigned long i = 0, t = 0;
  while (i < n)
  {
    t += l->items[i];
    i++;
  }
  return t;
}

int main()
{
  struct L a;
  a.items[0] = 1;
  a.items[1] = 2;
  a.items[2] = 3;
  a.items[3] = 4;
  a.size = 4;

  assert(total(&a, via_ternary(&a)) == 77);
  assert(total(&a, via_neq(&a)) == 10);
  assert(total(&a, via_early_return(&a)) == 10);
  assert(total(&a, (&a) ? a.size : 0) == 10);
  return 0;
}
