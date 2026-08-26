// esbmc/esbmc#7025: the base's own padding is named by component index, so it
// never matches the derived's; the displacement must be computed from the
// named members alone.
#include <cassert>

struct V
{
  int v;
};
struct A
{
  int a;
};
struct B : virtual V
{
  char c;
  int i;
  void put(char x)
  {
    c = x;
  }
};
struct D : A, B
{
};

int main()
{
  D d;
  d.put('A');
  assert(d.c == 'A');
  return 0;
}
