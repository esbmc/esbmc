// Based on ESBMC issue #589

#include <assert.h>
#include <string.h>

typedef struct str {
  unsigned short field1 : 4;
  unsigned short field2 : 4;
} str;

int main(void) {
  str array;
  str array_set[2];
  memset(&array, 0, sizeof(str));
  array_set[1] = array;
  array_set[1].field2 = 1;
  assert(*(unsigned short *)&array == *(unsigned short *)&array_set[1]);
  return 0;
}
