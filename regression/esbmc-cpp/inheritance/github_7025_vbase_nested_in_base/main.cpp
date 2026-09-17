// esbmc/esbmc#7025 residual: Mid keeps the nested "@base@" layout while D is
// flattened, so no member name is shared between the two and the upcast to
// Mid is left unadjusted.
#include <cassert>

struct V
{
  int v;
};
struct Base0
{
  int x;
};
struct Mid : Base0
{
  int m;
  void set(int k)
  {
    m = k;
  }
};
struct Vb : virtual V
{
  int q;
};
struct D : Vb, Mid
{
};

int main()
{
  D d;
  Mid *pm = &d;
  pm->set(7);
  assert(d.m == 7);
  return 0;
}
