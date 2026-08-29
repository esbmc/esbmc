#include <cstdlib>
#include <cassert>

int main()
{
  std::div_t d = std::div(7, 2);
  assert(d.quot == 3);
  assert(d.rem == 1);

  std::ldiv_t l = std::ldiv(-9L, 4L);
  assert(l.quot == -2L);
  assert(l.rem == -1L);

  std::lldiv_t ll = std::lldiv(9LL, 4LL);
  assert(ll.quot == 2LL);
  assert(ll.rem == 1LL);
  return 0;
}
