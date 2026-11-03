// A side-effecting placement-new address is evaluated exactly once, before
// the initializer ([expr.new]/19).
#include <cassert>
#include <new>
int slots[2];
int calls = 0;
void *slot()
{
  return &slots[calls++];
}
int main()
{
  ::new (slot()) int(calls);
  assert(calls != 1);
  return 0;
}
