// github #6494: clang resolves `delete p` to the C++14 sized form whenever one
// is declared, and both sized forms are declared implicitly -- so a program
// that replaces only operator delete(void *), by far the most common shape,
// resolved to a form it never defined and was left on the built-in path with
// its replacement never called. [new.delete.single]: the default sized
// operator delete calls operator delete(ptr).
#include <cstddef>
#include <cassert>

static char pool[64];
static int news = 0;
static int deletes = 0;

void *operator new(size_t)
{
  news++;
  return pool;
}
void operator delete(void *) noexcept
{
  deletes++;
}

int main()
{
  int *p = new int(1);
  assert(*p == 1);
  assert(news == 1);
  delete p;
  assert(deletes == 1);
  return 0;
}
