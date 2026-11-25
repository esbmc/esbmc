#include <assert.h>
#include <stdio.h>

struct s
{
  int p0;
  char hh;
  int p1;
  short h;
  int p2;
  long long ll;
  int p3;
  unsigned long z;
  int p4;
  float f;
  int p5;
  double d;
  int p6;
  long double L;
  int p7;
  void *p;
  int p8;
} g = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};

int main()
{
  sscanf(
    "1 2 3 4 5 6 7 0",
    "%hhd %hd %lld %zu %f %lf %Lf %p",
    &g.hh,
    &g.h,
    &g.ll,
    &g.z,
    &g.f,
    &g.d,
    &g.L,
    &g.p);
  assert(g.p0 == 1 && g.p1 == 1 && g.p2 == 1 && g.p3 == 1);
  assert(g.p4 == 1 && g.p5 == 1 && g.p6 == 1 && g.p7 == 1 && g.p8 == 1);
  return 0;
}
