#include <assert.h>
#include <string.h>

typedef struct str {
  unsigned char field1;
  unsigned char field2;
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
