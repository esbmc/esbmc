#include <stdlib.h>
#include <assert.h>

// Anchor malloc(-1): the magnitude-1 case would be misclassified as a
// 1-byte allocation if symex_mem ever checked v == 1 before the
// negative-size branch (to_uint64() drops the sign).
int main()
{
  void *b = (void *)malloc(-1);
  assert(b == NULL);
}
