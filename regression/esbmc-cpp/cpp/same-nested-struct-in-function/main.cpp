#include <cassert>

struct a
{
  a()
  {
    struct _Guard
    {
      char faaa = 1;
      struct bar
      {
      } bar;
    } __guard;
    __guard.faaa;
    assert(__guard.faaa == 1);
  }
};
struct b
{
  b()
  {
    struct _Guard
    {
      int _M_guarded = 2;
      struct bar
      {
        double aaa = 3;
      } bar;
    } __guard;
    __guard._M_guarded;
    assert(__guard._M_guarded == 2);
    assert(__guard.bar.aaa == 3);
  }
};
struct c
{
  c()
  {
    struct _Guard
    {
      double ffff = 4;
      struct bar
      {
        int aaa = 5;
      } bar;
    } __guard;
    __guard.ffff;
    assert(__guard.ffff == 4);
    assert(__guard.bar.aaa == 5);
  }
};
int main()
{
  a a;
  b b;
  c c;
  return 0;
}
