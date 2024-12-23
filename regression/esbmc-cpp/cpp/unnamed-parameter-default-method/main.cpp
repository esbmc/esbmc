
#include <cassert>

struct Foo
{
  Foo &operator=(Foo &) = default;
  int _M_payload;
};
int main()
{
  Foo tmp;
  tmp._M_payload = 22;
  Foo distinctValues;
  distinctValues = tmp;
  assert(distinctValues._M_payload == 22);
}
