#include <assert.h>

int main()
{
  int other = 0;
  int &other_ref = other;
  assert(++other_ref == 1);
}
