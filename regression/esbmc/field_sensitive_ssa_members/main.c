/* A bool, a float and an array field split; the block-scope s dies each
 * iteration, and the global g stays whole. */
#include <assert.h>
#include <stdbool.h>

struct S
{
  bool b;
  float f;
  int a[2];
};

struct G
{
  int x;
} g;

int nondet_int();

int main()
{
  int sum = 0;
  for (int i = 0; i < 2; i++)
  {
    struct S s;
    s.b = i;
    s.f = 1.5f;
    s.a[0] = i;
    s.a[1] = nondet_int();
    sum += s.a[0] + (s.b ? 1 : 0) + (int)s.f + g.x;
  }
  assert(sum == 4);
  return 0;
}
