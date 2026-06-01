// Negative companion to github_4267: when the struct is NOT packed,
// a misaligned pointer access must still produce an alignment VCC.
#include <cstdint>

struct S
{
  uint8_t a;
  uint32_t b;
};

int main()
{
  alignas(4) char buf[16];
  // Force an unaligned 4-byte access to 'b'.
  S *p = reinterpret_cast<S *>(buf + 1);
  p->b = 0;
  return 0;
}
