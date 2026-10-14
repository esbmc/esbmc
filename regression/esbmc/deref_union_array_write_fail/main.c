#include <assert.h>

/* A union whose first member is an array, stored through a pointer to a larger
 * byte buffer: only the first 4 bytes change. */
union U
{
  unsigned char a[4];
  int i;
};

int main(void)
{
  unsigned char b[8] = {10, 11, 12, 13, 14, 15, 16, 17};
  union U v;
  v.a[0] = 1;
  v.a[1] = 2;
  v.a[2] = 3;
  v.a[3] = 4;
  *(union U *)b = v;
  assert(b[5] == 4);
  return 0;
}
