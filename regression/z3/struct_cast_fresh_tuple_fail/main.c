#include <assert.h>

struct H
{
  int t;
};

struct A
{
  int t;
} a = {1}, b;

int main()
{
  struct H *p = nondet_bool() ? (struct H *)&a : (struct H *)&b;
  struct H h = *p;
  assert(h.t == 1);
}
