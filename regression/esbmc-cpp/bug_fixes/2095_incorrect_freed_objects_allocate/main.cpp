#include <stddef.h>
#include <stdlib.h>
struct
{
  float *allocate()
  {
    return static_cast<float *>(malloc(sizeof(float)));
  }
} a;
float *allocate()
{
  return a.allocate();
}
float *b()
{
  return allocate();
}
int main()
{
  float *x = b();
  free(x);
  return 0;
}
