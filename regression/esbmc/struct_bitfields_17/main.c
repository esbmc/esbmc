// Based on ESBMC issue #589

#include <assert.h>

/*
$ gcc -fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer main.c -o main
$ ./main
main.c:21:3: runtime error: load of misaligned address 0x7ffe421e1bc6 for type 'unsigned int', which requires 4 byte alignment
0x7ffe421e1bc6: note: pointer points here
 00 00 30 00 00 00  00 00 00 00 00 00 00 00  00 00 00 00 00 00 02 00  00 00 00 00 54 06 ea a7  00 00
*/

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
