#include <assert.h>

/* A union whose first member is an array, read through a pointer to a larger
 * byte buffer: the array member takes the first 4 bytes, not the buffer. */
union U
{
  unsigned char a[4];
  int i;
};

int main(void)
{
  unsigned char b[8] = {10, 11, 12, 13, 14, 15, 16, 17};
  union U u = *(union U *)b;
  assert(u.a[3] == 14);
  return 0;
}
