#include <cassert>

struct a
{
  int f = 4;
  a &e()
  {
    return *this;
  }
} ag;
int main()
{
  a &the_kind = ag.e();
  assert(the_kind.f == 444);
}
