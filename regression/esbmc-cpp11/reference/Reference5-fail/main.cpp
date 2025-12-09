#include <assert.h>

float dd = 50.0f;
struct f
{
  float &c = dd;
};
int main()
{
  f i;
  i.c = 20.0f;
  assert(i.c == 0.0f);
}
