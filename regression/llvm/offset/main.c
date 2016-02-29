#include <assert.h>

struct S {
  float f;
  double d;
};

struct T {
  int i;
  struct S s[10];
};

int main()
{
  struct abc1 { int a, b, c; };
  union abc2 { int a, b, c; };

  assert(__builtin_offsetof(struct abc1, a) == 0); // always, because there's no padding before a.
  assert(__builtin_offsetof(struct abc1, b) == 4); // here, on my system
  assert(__builtin_offsetof(union abc2, a) == __builtin_offsetof(union abc2, b)); // (members overlap)

  assert(__builtin_offsetof(struct T, s[2].d) == 48);
  return 0;
}
