// A constructor left by an exception destroys the bases and members it has
// already built, newest first ([except.ctor]/3).
#include <cassert>

int log_ = 0;

struct C
{
  int v;
  C(int x) : v(x)
  {
    if (x == 0)
      throw 1;
  }
  ~C()
  {
    log_ = log_ * 10 + v;
  }
};

struct Members
{
  C a;
  int k;
  C b;
  C c;
  Members() : a(1), b(2), c(0)
  {
  }
};

struct Complete
{
  C a;
  C b;
  Complete() : a(6), b(7)
  {
  }
};

struct Base
{
  C x{3};
};

struct Body : Base
{
  C y;
  Body() : y(4)
  {
    throw 2;
  }
};

int main()
{
  try
  {
    Members m;
  }
  catch (int)
  {
  }
  assert(log_ == 21);

  log_ = 0;
  try
  {
    Body b;
  }
  catch (int)
  {
  }
  assert(log_ == 43);

  log_ = 0;
  {
    Complete c;
  }
  assert(log_ == 76);
}
