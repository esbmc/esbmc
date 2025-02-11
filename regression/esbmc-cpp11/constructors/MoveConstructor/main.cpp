#include <assert.h>
#include <utility>

struct MyStruct
{
  int value;
};

int main()
{
  MyStruct a = {10};

  // move ctor
  MyStruct c(std::move(a));

  assert(c.value == 10);
}
