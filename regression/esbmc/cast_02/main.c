#include <assert.h>
#include <string.h>

/*
$ clang -fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer main.c -o main
$ ./main
main.c:24:3: runtime error: load of misaligned address 0x7fffb0eb98a1 for type 'unsigned int', which requires 4 byte alignment
0x7fffb0eb98a1: note: pointer points here
 00 00 00  01 23 45 67 89 ab cd ef  ff 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  00 00 00 00 00
              ^
SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior main.c:24:3 in
main.c:24:3: runtime error: load of misaligned address 0x7fffb0eb98a5 for type 'unsigned int', which requires 4 byte alignment
0x7fffb0eb98a5: note: pointer points here
 23 45 67 89 ab cd ef  ff 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  28
             ^
SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior main.c:24:3 in
*/

int main(void) {
  unsigned char char_array[] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFF};
  unsigned short short_array[4];
  memcpy(&short_array, char_array, sizeof(short_array));
  assert(*(unsigned int *)&char_array == *(unsigned int *)&short_array);
  
  unsigned int* ptr = (unsigned int *)&char_array[1];
  assert(*ptr == 0x89674523 && *(++ptr) == 0xFFEFCDAB);
  return 0;
}

