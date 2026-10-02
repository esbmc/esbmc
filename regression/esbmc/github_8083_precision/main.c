#include <stdio.h>
#include <string.h>
#include <assert.h>

/* What these calls write is a known variable, or nothing: scanf's format,
   fscanf's stream and memcmp's operands are read only. The interval of y
   survives each call and the analysis still proves y == 3. */
int main()
{
  int x = 5, y = 3, z = 4;
  int a[2] = {0, 0};
  scanf("%d", &x);
  assert(y == 3);
  fscanf(stdin, "%d", &a[1]);
  assert(y == 3);
  if (memcmp(&x, &z, sizeof x) == 0)
    x = 0;
  assert(y == 3);
  return 0;
}
