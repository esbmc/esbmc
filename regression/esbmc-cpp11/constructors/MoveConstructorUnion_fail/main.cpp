#include <assert.h>
#include <utility>

union MyUnion
{
  int value;
};

int main()
{
  MyUnion a = {10};

  // move ctor
  MyUnion c(std::move(a));

  assert(c.value == 0); //should be 10
}
