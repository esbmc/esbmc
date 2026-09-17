/* A loop bounded through a null-guarded address-of must cost the same at any
   --unwind. Before the guard folded this was 20 loop unwindings and 45 VCCs at
   --unwind 20, and 165 VCCs at --unwind 80; it is now 4 unwindings and 12 VCCs
   at both bounds. */
#include <assert.h>

struct L
{
  int items[10];
  unsigned long size;
};

static unsigned long lsize(const struct L *l)
{
  return l ? l->size : 0;
}

int main()
{
  struct L a;
  a.items[0] = 1;
  a.items[1] = 2;
  a.items[2] = 3;
  a.items[3] = 4;
  a.size = 4;

  unsigned long n = lsize(&a);
  unsigned long i = 0, t = 0;
  while (i < n)
  {
    t += a.items[i];
    i++;
  }
  assert(t == 10);
  return 0;
}
