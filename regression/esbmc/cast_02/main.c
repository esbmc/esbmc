#include <assert.h>
#include <string.h>

int main(void) {
  unsigned char char_array[] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFF};
  unsigned short short_array[4];
  memcpy(&short_array, char_array, sizeof(short_array));
  assert(*(unsigned int *)&char_array == *(unsigned int *)&short_array);
  
  unsigned int* ptr = (unsigned int *)&char_array[1];
  assert(*ptr == 0x89674523 && *(++ptr) == 0xFFEFCDAB);
  return 0;
}

