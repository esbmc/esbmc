#include <cassert>
#include <new>

int main()
{
  void *foo = operator new(sizeof(int));

  int *intPtr = static_cast<int *>(foo);
  *intPtr = 42;

  assert(*intPtr == 42);

  operator delete(foo);

  return 0;
}
