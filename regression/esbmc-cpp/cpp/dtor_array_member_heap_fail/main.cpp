// Negative counterpart of dtor_array_member_heap: exactly one ~B runs per
// array element, so g is 3 after destroying a[3]. Asserting g == 4 must fail,
// pinning the element count (and catching any spurious extra destructor call).
#include <cassert>

int g;

struct B
{
  int x;
  ~B()
  {
    g++;
  }
};

struct C
{
  B a[3];
};

int main()
{
  C *c = new C();
  delete c;
  assert(g == 4);
  return 0;
}
