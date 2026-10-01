/* Reading a union through its bytes flattens a vector member lane by lane. */
#include <assert.h>
typedef int v4i __attribute__((vector_size(16)));
union U
{
  v4i a[2];
  char c[32];
};
int main()
{
  union U u;
  u.a[1] = (v4i){1, 2, 3, 4};
  assert(u.c[16] == 1 && u.c[17] == 0 && u.c[20] == 2 && u.c[28] == 4);
  return 0;
}
