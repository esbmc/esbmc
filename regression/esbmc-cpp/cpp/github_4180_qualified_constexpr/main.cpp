#include <cstdint>
#include <cassert>

namespace outer
{
namespace inner
{
constexpr unsigned long Value = 64;
} // namespace inner
} // namespace outer

namespace outer
{
struct Header
{
  uint8_t a;
};
// Trigger: qualified `inner::Value` used in a sibling-namespace constexpr
// initialiser involving sizeof.
constexpr unsigned long DataLen = inner::Value + sizeof(Header);
} // namespace outer

int main()
{
  assert(outer::DataLen == 65);
  return (int)outer::DataLen;
}
