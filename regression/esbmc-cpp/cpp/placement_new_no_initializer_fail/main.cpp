// A placement new with no initializer still evaluates its address.
#include <cassert>
#include <new>
int slots[4];
int calls = 0;
void *slot()
{
  return &slots[calls++];
}
int main()
{
  ::new (slot()) int;
  assert(calls != 1);
  return 0;
}
