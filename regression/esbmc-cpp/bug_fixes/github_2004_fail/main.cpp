#include <cassert>
#include <new>

int main()
{
  void *foo = operator new(sizeof(int));

  // allocated memory is not enough
  // int 4; double 8
  double *intPtr = static_cast<double *>(foo);
  *intPtr = 42;

  assert(*intPtr == 42);

  operator delete(foo);

  return 0;
}
