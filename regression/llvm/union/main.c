//#include <assert.h>
//#include <stdio.h>

union AB 
{
  int a;
  char b;
} ab, XXX = {1};

union AB1
{
  int a1;
  float b1;
};

union T {
  int i;
  union AB s[2];
} t = {1, { 1, 3 } };

union AB1 YYY = {1};

void func(union AB foo)
{
  assert(foo.a == 1);
  assert(foo.b == 1);
}

int main()
{
  union AB2
  {
    int a2;
    float b2;
  };

  union AB3
  {
    int a3;
    float b3;
  } ab3 = {1};

  union AB foo1;
  union AB foo2 = {1};
  func(foo2);

  assert(ab.a == 0);
  assert(ab.a == 0);

  return 0;
}
