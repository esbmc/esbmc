/* github #8711: a NaN copied into a float keeps being a NaN */
#include <assert.h>
#include <math.h>
#include <string.h>
struct S
{
  int a;
  float f;
};
int main(void)
{
  float src[2] = {1.0f, NAN};
  struct S s;
  memcpy(&s.f, &src[1], sizeof(float));
  double d = 3.0;
  float g;
  memcpy(&g, (char *)&d + 4, sizeof g);
  assert(s.f != s.f);
  unsigned u;
  memcpy(&u, (char *)&d + 4, 4);
  float h;
  memcpy(&h, &u, 4);
  assert(g == h);
  return 0;
}
