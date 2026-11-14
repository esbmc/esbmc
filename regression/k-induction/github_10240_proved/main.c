#include <assert.h>

void f(float beta, int n)
{
  float s = 0.0f;
  for (int i = 0; i < n; ++i)
    s += 1.0f;
  assert(beta == 0.0f);
}

int main()
{
  f(0.0f, 1000);
  return 0;
}
