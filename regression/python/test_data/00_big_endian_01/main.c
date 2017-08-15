#include <stdio.h>
#include <assert.h>

int main() {
  float i=0x0102;
  char *p=(char *)&i;
//  printf("Bytes of i: %d, %d, %d, %d\n",
//         p[0], p[1], p[2], p[3]);

  printf("Bytes of i: %d, %d\n", p[0], p[1]);
  assert(p[0]==0);
  assert(p[1]==0);
//  assert(0);
} 
