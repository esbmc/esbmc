#include <cassert>

class c
{
public:
  virtual int foo()
  {
    return 22;
  }
};
class C : public c
{
public:
  C();
  virtual int foo()
  {
    return 33;
  }
};

C::C()
{
}
int main()
{
  C b;
  C *bp = &b;
  assert(bp->foo() == 33);
}
