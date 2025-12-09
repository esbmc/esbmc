
#include <cassert>
struct a
{
  double d;
  a(double double_) : d(double_)
  {
  }
  a() : a(2.0)
  {
  }
};

int main()
{
  a second(5.0);
  a third;
  assert(second.d == 5.0);
  assert(third.d == 200.0);
}