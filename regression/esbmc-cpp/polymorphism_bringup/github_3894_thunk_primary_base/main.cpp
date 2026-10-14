// esbmc/esbmc#3894: the thunk that adapts a Base* receiver to the overriding
// method's Derived* must take its displacement from ESBMC's own layout. B is
// D's primary base under the Itanium ABI (the only polymorphic one), so
// clang's getBaseClassOffset(B) is 0 while ESBMC places @base@tag-B after
// @base@tag-A -- the thunk then re-based by 0 and read past the object.
#include <cassert>

struct A
{
  int a;
};
struct B
{
  virtual ~B()
  {
  }
  virtual int f()
  {
    return 0;
  }
  int b;
};
struct D : A, B
{
  int d;
  int f() override
  {
    return b + d;
  }
};

int main()
{
  D o;
  o.a = 1;
  o.b = 2;
  o.d = 3;
  B *pb = &o;
  assert(pb->f() == 5);
  return 0;
}
