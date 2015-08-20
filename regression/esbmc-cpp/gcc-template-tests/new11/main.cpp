#include <cassert>
#include <cstddef>
// PR c++/54984
// { dg-do run }
int n = 1;

void* operator new(size_t)
{
  n = -1;
  return &n;
}

template <class T>
struct Foo
{
  Foo()
  : x(new int)
  {
    if (*x != -1)
	assert(0);
  }

  int* x;
};

int main()
{
  Foo<float> foo;
}
