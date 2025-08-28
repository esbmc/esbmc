#include <stdlib.h>
#include <string.h>
#include <assert.h>

int main(void)
{
  char *result = getenv("NONEXISTENT_VAR_12345");
  // This should fail if getenv returns NULL
  size_t len = strlen(result);
  assert(len >= 0);
  return 0;
}
