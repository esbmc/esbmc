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
  assert(u.c[20] == 1);
  return 0;
}
