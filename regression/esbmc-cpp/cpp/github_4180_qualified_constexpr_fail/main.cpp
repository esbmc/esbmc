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
constexpr unsigned long DataLen = inner::Value + sizeof(Header);
} // namespace outer

int main()
{
  // outer::DataLen is 65 (64 + sizeof(Header)==1); deliberately wrong claim.
  assert(outer::DataLen == 64);
  return (int)outer::DataLen;
}
