#include <assert.h>

union MyUnion
{
  int value;
};

int main()
{
  MyUnion a = {10};

  // copy ctor
  MyUnion c(a);

  assert(c.value == 0); //should be 10
}
