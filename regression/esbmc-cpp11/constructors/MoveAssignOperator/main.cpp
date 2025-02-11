#include <assert.h>
#include <utility>

struct MyStruct
{
  int value;
};

int main()
{
  MyStruct a = {10};

  MyStruct b = {5};

  // move assign
  b = std::move(a);

  assert(b.value == 10);
}
