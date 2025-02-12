#include <stddef.h>
#include <stdlib.h>
struct
{
  float *alloca$te()
  {
    return static_cast<float *>(malloc(sizeof(float)));
  }
} a;
float *alloca$te()
{
  return a.alloca$te();
}
float *b()
{
  return alloca$te();
}
int main()
{
  float *x = b();
  free(x);
  free(x);
  return 0;
}
