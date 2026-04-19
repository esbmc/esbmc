#include <assert.h>
#include <stdlib.h>

void unknown_method(int *a);
int a = 1;

int main()
{
  int *p = &a;

  unknown_method(p);

  *p; // access invalid pointer
}
