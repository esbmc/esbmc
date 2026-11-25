#include <assert.h>
#include <string.h>
int main(void)
{
  char d[4];
  memcpy(d, 0, 0);
  for (unsigned x = 0; x < 10; x++)
    assert(x < 10);
  return 0;
}
