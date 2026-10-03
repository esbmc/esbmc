#include <cassert>
#include <new>

struct C
{
  int v;
  C() : v(7)
  {
  }
};

alignas(C) unsigned char buf[3 * sizeof(C)];
int calls;

unsigned char *slot()
{
  ++calls;
  return buf;
}

int main()
{
  int *a = new (buf) int[2]{1, 2};
  assert((void *)a == (void *)buf);
  assert(reinterpret_cast<int *>(buf)[1] == 2);

  unsigned n = 3;
  C *p = new (slot()) C[n];
  assert(calls == 1);
  assert(p == reinterpret_cast<C *>(buf));
  assert(reinterpret_cast<C *>(buf)[2].v == 7);
  return 0;
}
