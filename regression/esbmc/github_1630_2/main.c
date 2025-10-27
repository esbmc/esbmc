#include <string.h>
#include <assert.h>
#include <stdio.h>

#define SMALL 200

int main() {
  {
    char src[SMALL] = "HelloWorld";
    char dst[SMALL];
    strncpy(dst, src, SMALL);
  }

  {
    char src[SMALL] = "Short";
    char dst[SMALL];
    strncpy(dst, src, SMALL);
    assert(dst[5] == '\0');
    for (int i = 6; i < SMALL; i++)
      assert(dst[i] == '\0');
  }

  {
    char src[SMALL] = "ThisStringIsTooLong";
    char dst[8];
    strncpy(dst, src, sizeof(dst));
    assert(strncmp(dst, "ThisStr", 7) == 0);
  }
  
  return 0;
}

