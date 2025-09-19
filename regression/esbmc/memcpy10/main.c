#include <string.h>
#include <assert.h>

int main() {
  char src[] = {'a','b','c','d'};
  char dst[4];
  memcpy(dst, src, 4);
  assert(dst[0] == 'a' && dst[3] == 'd');
  return 0;
}
