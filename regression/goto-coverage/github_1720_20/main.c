#include <stdbool.h>
#include <assert.h>

int main()
{
  bool a = true;
  bool b = false;

  assert(a != 0);
  if (a ? b ? 1 : 0 : a == b)
    ;
}