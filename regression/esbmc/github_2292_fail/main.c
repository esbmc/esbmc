#include <assert.h>
typedef struct {
  _Bool (*a)();
} c;
void *e;
c f[1];
c *g = f;
int i;
_Bool b() { return 1; }
void d();
int main() {
  f[0] = (c){b};
  if (!g[i].a())
    e = d;
  assert(e);
  return 0;
}
