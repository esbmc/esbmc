// The pool forgets to advance, so both objects share its first slot.
#include <cassert>
#include <cstddef>

struct Pool
{
  alignas(8) unsigned char buf[64];
  std::size_t used = 0;
};

void *operator new(std::size_t n, Pool &p)
{
  return p.buf + p.used;
}

int main()
{
  Pool pool;
  int *a = new (pool) int(1);
  int *b = new (pool) int(2);
  assert(*a == 1);
  return 0;
}
