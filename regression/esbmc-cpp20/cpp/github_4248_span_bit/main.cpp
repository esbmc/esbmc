// github.com/esbmc/esbmc/issues/4248 — bundled <span> must transitively
// expose std::bit_cast, matching libc++/libstdc++.

#include <span>
#include <cstdint>
#include <cassert>

void process(std::span<const uint8_t> s)
{
  const uint8_t *p = std::bit_cast<const uint8_t *>(s.data());
  assert(p == s.data());
}

int main()
{
  uint8_t buf[4] = {1, 2, 3, 4};
  process(std::span<const uint8_t>(buf, 4));
  return 0;
}
