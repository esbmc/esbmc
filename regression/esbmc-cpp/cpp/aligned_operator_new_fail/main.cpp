#include <cassert>
#include <cstddef>
#include <new>

alignas(64) static unsigned char pool[64];

void *operator new(std::size_t, std::align_val_t)
{
  return pool;
}
void operator delete(void *, std::align_val_t) noexcept
{
}

struct alignas(64) A
{
  int x;
};

int main()
{
  A *a = new A;
  A *b = new A;
  assert(a != b);
  return 0;
}
