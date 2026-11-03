// Each evaluation of a placement new in a mem-initializer has its own
// address, including a recursive one.
#include <cassert>
#include <new>
int slots[4];
int calls = 0;
void *slot()
{
  return &slots[calls++];
}
struct N
{
  int *p;
  N(int k) : p(::new (slot()) int(k ? *N(k - 1).p + 10 : 0))
  {
  }
};
int main()
{
  N n(1);
  assert(slots[0] == 10 && slots[1] == 0);
  return 0;
}
