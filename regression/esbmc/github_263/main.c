#include <assert.h>

int MAX = 3;

int main()
{
  int a[MAX][MAX];
  assert(a[0] < a[MAX-1]);
}

