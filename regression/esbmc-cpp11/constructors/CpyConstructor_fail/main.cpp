#include <assert.h>

struct MyStruct
{
  int value;
};

int main()
{
  MyStruct a = {10};

  // copy ctor
  MyStruct c(a);

  assert(c.value == 0); //should be 10
}
