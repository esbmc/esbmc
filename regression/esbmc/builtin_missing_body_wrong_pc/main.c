#include <assert.h>
extern void __builtin();

int main()
{
  __builtin();

  assert(0);
}