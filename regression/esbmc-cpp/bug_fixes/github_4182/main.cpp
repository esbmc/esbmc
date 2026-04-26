#include <cassert>

namespace ns {
struct Packet
{
  int x;
};
}  // namespace ns

using ns::Packet;  // using-declaration of a class type

int main()
{
  Packet pkt{};
  assert(pkt.x == 0);
  return pkt.x;
}
