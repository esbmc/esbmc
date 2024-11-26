#include <cassert>

struct a
{
  a() : b(22)
  {
  }
  struct
  {
    struct
    {
      int b;
    };
  };
};
int main()
{
  a a;
  assert(a.b == 22);
}
