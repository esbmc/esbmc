#include <cassert>
int seen = 0;
struct B
{
  virtual int f()
  {
    return 1;
  }
  virtual ~B() throw()
  {
    seen = f();
  }
};
struct D : B
{
  int f()
  {
    return 2;
  }
};
int main()
{
  {
    D d;
  }
  assert(seen == 2); // wrong per [class.cdtor]/4: B's dtor sees B::f
  return 0;
}
