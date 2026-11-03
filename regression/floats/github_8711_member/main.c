/* github #8711 */
#include <assert.h>
#include <string.h>
struct S
{
  int a;
  float f;
  double d;
};
int main(void)
{
  const unsigned char b4[4] = {0x00, 0x00, 0x80, 0x3f};       /* 1.0f */
  const unsigned char b8[8] = {0, 0, 0, 0, 0, 0, 0xf0, 0x3f}; /* 1.0 */
  struct S s;
  float arr[3];
  memcpy(&s.f, b4, sizeof s.f);
  memcpy((char *)&s + 8, b8, 8);
  memcpy(&arr[1], b4, 4);
  assert(s.f == 1.0f);
  assert(s.d == 1.0);
  assert(arr[1] == 1.0f);
  return 0;
}
