// A placement-new address that is a call constructs in the buffer, not in a
// fresh allocation ([expr.new]/11).
#include <cassert>
#include <memory>
#include <new>
int main()
{
  alignas(int) unsigned char buf[sizeof(int)] = {};
  int *p = ::new (static_cast<void *>(std::addressof(buf))) int(42);
  assert(*reinterpret_cast<int *>(buf) != 42);
  return 0;
}
