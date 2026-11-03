#include <assert.h>
#include <stdio.h>

int main(void)
{
  // Each argument is converted to the conversion's type before printing; a
  // length taken from the argument's own type would make this path vacuous.
  int a = printf("%d", 4294967295u);
  int b = printf("%lu", -1);
  int c = printf("%d", (long long)5000000000);
  assert(0);
  return 0;
}
