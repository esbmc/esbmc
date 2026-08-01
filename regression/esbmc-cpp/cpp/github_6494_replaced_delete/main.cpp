// github #6494, the false-alarm direction: a replaced operator delete was never
// called, so state it maintains never changed and this correct program was
// reported as failing. Covers the C++14 sized form too -- clang selects
// operator delete(void *, size_t) here, and both increment the counter.
#include <cstddef>
#include <cassert>

static int deletes = 0;

void operator delete(void *) noexcept
{
  deletes++;
}
void operator delete(void *, size_t) noexcept
{
  deletes++;
}

int main()
{
  int *p = new int(1);
  assert(*p == 1);
  delete p;
  assert(deletes == 1);
  return 0;
}
