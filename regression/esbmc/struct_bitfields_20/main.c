// Based on ESBMC issue #575

#include <assert.h>
#include <stdlib.h>

typedef struct str {
  unsigned : 1;
  unsigned : 2;
  unsigned field1 : 8;
} str;

int main(void) {

  str array, *ptr = &array;
  int i = 0;
  ptr[i].field1 = i;
  unsigned int n = ptr->field1;
  assert(n == i);

  return 0;
}
