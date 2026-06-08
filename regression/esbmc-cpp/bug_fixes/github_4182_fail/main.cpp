#include <cassert>

namespace ns {
struct Packet
{
  int x;
};
}  // namespace ns

using ns::Packet;

int main()
{
  Packet pkt{};
  assert(pkt.x == 1);
  return pkt.x;
}
