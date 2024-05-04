#include <assert.h>

struct b
{
  int x;
  b(float a, int &d)
  {
    int temp = d;
    x = temp;
  }
};
struct e : b
{
  template <typename... f>
  e(f... g) : b(g...)
  {
  }
};
float h;
int main()
{
  e ee = e(h, 44);
  assert(ee.x = 0);
}