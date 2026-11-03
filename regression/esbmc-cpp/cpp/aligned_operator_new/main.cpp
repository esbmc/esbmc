#include <cassert>
#include <cstddef>
#include <new>

alignas(64) static unsigned char pool[512];
int news = 0, deletes = 0;
std::size_t last_size = 0, last_align = 0;

void *operator new(std::size_t n, std::align_val_t a)
{
  news++;
  last_size = n;
  last_align = static_cast<std::size_t>(a);
  return pool;
}
void *operator new[](std::size_t n, std::align_val_t a)
{
  news++;
  last_size = n;
  last_align = static_cast<std::size_t>(a);
  return pool + 128;
}
void operator delete(void *p, std::align_val_t a) noexcept
{
  assert(p == pool && static_cast<std::size_t>(a) == 64);
  deletes++;
}
void operator delete[](void *p, std::align_val_t a) noexcept
{
  assert(p == pool + 128 && static_cast<std::size_t>(a) == 64);
  deletes++;
}

struct alignas(64) A
{
  int x[32];
};

int main()
{
  A *a = new A;
  assert(news == 1 && last_size == sizeof(A) && last_align == 64);
  assert(static_cast<void *>(a) == pool);
  delete a;
  assert(deletes == 1);

  A *b = new A[2];
  assert(news == 2 && last_size == 2 * sizeof(A) && last_align == 64);
  delete[] b;
  assert(deletes == 2);
  return 0;
}
