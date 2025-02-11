#include <assert.h>

typedef union
{
  int a;
} b;
struct c d;
struct c
{
  b a;
} e(b f);

void g(struct c *f)
{
  f->a.a = 2;
  e(f->a);
}
int main()
{
  g(&d);
  assert(d.a.a == 1);
}

