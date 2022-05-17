// Based on ESBMC issue #589

#include <assert.h>

typedef struct str2 {
  unsigned short field1 : 4;
  unsigned short field2 : 4;
} str2;

typedef struct str {
  unsigned short field1 : 4;
  unsigned short field2 : 4;
  str2 arr[10];
} str;

int main(void) {
  assert(sizeof(str) == 22);
  str array;
  str array_set[2];
  array_set[1] = array;
  assert(*(unsigned int *)&array == *(unsigned int *)&array_set[1]);
  return 0;
}
