#include <cstdint>
#include <cassert>

struct S { uint8_t a; uint16_t b; };
class C { public: C() = default; S s{}; };

int main()
{
  C c{};
  assert(c.s.a == 0 && c.s.b == 0);
}
