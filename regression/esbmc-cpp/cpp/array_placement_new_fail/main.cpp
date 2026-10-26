#include <cassert>
#include <new>

alignas(int) unsigned char buf[2 * sizeof(int)];

int main()
{
  new (buf) int[2]{1, 2};
  assert(reinterpret_cast<int *>(buf)[1] != 2);
  return 0;
}
