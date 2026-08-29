#include <assert.h>

extern int undefined_extern;

int main(void)
{
  assert(undefined_extern == 0);
  return 0;
}
