#include <assert.h>

struct P
{
  int x;
};

int calls = 0;
int next(void)
{
  return ++calls;
}

int pick(int k)
{
  switch (k)
  {
  case 0:
    return (struct P){5}.x;
  }
  return 0;
}

int main(void)
{
  int s = 0;
  for (int i = 0; i < 3; i++)
    s += (struct P){i}.x;
  assert(s == 3);

  int c = 0, t = 0;
  if (c)
    t = (struct P){next()}.x;
  assert(calls == 0 && t == 0);

  assert(pick(0) == 5);
  return 0;
}
