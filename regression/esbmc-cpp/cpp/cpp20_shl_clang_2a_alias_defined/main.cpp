// Same expression compiled with --std c++2a (Clang's pre-finalisation
// alias for C++20). Confirms is_cxx20_or_later() recognises the alias.
#include <cstdint>

extern "C" unsigned char nondet_uchar();

int main()
{
  uint8_t  b = nondet_uchar();
  uint32_t r = static_cast<uint32_t>(b << 24);
  return r == 0 ? 0 : 1;
}
