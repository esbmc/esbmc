#include <assert.h>
 
static_assert(sizeof(long) == 4, "Code relies on int being exactly 4 bytes");
 
int main(void)
{
  static_assert(sizeof(long) == 4, "Code relies on int being exactly 4 bytes");

  return 0;
}
