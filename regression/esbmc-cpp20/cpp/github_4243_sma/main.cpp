#include <array>
#include <cstdint>

struct SmaFilter
{
  std::array<int32_t, 4> _buffer{};
  int32_t _sum{};

  void update(int32_t x)
  {
    _sum += x - _buffer[0];
  }
};

int main()
{
  SmaFilter f;
  f.update(53248);
  return 0;
}
