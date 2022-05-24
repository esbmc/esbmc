#include <assert.h>
#include <string.h>

int *p = &(int){5};

int main()
{
  assert(p);
  assert(*p == 5);
  ++*p;
  assert(*p == 6);
}
