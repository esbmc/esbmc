// A placement new under a case label, reached recursively, keeps each
// frame's address.
#include <cassert>
#include <new>
int slots[4];
int calls = 0;
void *slot()
{
  return &slots[calls++];
}
int f(int k)
{
  switch (k)
  {
  case 0:
    return 0;
  default:
    return *::new (slot()) int(f(k - 1) + 10);
  }
}
int main()
{
  int r = f(2);
  assert(r == 20 && slots[0] == 20);
  return 0;
}
