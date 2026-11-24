#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
int main()
{
  char *p = calloc(SIZE_MAX, 2);
  assert(p == NULL); // C23 7.24.3.2: nmemb * size wraps size_t
  return 0;
}
