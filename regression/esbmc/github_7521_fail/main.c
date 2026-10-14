#include <assert.h>

/* #7521: value_sett::assign walked every member of a struct on every
 * assignment, so symex cost scaled with struct width. The walk is now skipped
 * for pointer-free aggregates -- but it must still happen wherever a pointer
 * can hide, or the dereferences below lose their value sets and fail. */

struct inner
{
  int *ip;
};

struct wide
{
  int x;
  int *q;             /* direct pointer member */
  struct inner nest;  /* pointer reachable through a nested struct */
  int *arr[2];        /* pointer reachable through an array member */
  int p0;
  int p1;
  int p2;
  int p3;
  int p4;
  int p5;
  int p6;
  int p7;
  int p8;
  int p9;
  int p10;
  int p11;
  int p12;
  int p13;
  int p14;
  int p15;
  int p16;
  int p17;
  int p18;
  int p19;
  int p20;
  int p21;
  int p22;
  int p23;
  int p24;
  int p25;
  int p26;
  int p27;
  int p28;
  int p29;
  int p30;
  int p31;
  int p32;
  int p33;
  int p34;
  int p35;
  int p36;
  int p37;
  int p38;
  int p39;
  int p40;
  int p41;
  int p42;
  int p43;
  int p44;
  int p45;
  int p46;
  int p47;
  int p48;
  int p49;
  int p50;
  int p51;
  int p52;
  int p53;
  int p54;
  int p55;
  int p56;
  int p57;
  int p58;
  int p59;
  int p60;
  int p61;
  int p62;
  int p63;
};

struct plain
{
  int x;
  int p0;
  int p1;
  int p2;
  int p3;
  int p4;
  int p5;
  int p6;
  int p7;
  int p8;
  int p9;
  int p10;
  int p11;
  int p12;
  int p13;
  int p14;
  int p15;
  int p16;
  int p17;
  int p18;
  int p19;
  int p20;
  int p21;
  int p22;
  int p23;
  int p24;
  int p25;
  int p26;
  int p27;
  int p28;
  int p29;
  int p30;
  int p31;
  int p32;
  int p33;
  int p34;
  int p35;
  int p36;
  int p37;
  int p38;
  int p39;
  int p40;
  int p41;
  int p42;
  int p43;
  int p44;
  int p45;
  int p46;
  int p47;
  int p48;
  int p49;
  int p50;
  int p51;
  int p52;
  int p53;
  int p54;
  int p55;
  int p56;
  int p57;
  int p58;
  int p59;
  int p60;
  int p61;
  int p62;
  int p63;
};

int main()
{
  int a = 7, b = 8, c = 9, d = 10;
  struct wide w;
  struct plain p;

  w.x = 1;
  w.q = &a;
  w.nest.ip = &b;
  w.arr[0] = &c;
  w.arr[1] = &d;
  p.x = 1;

  for (int i = 0; i < 4; i++)
  {
    assert(*w.q == 7);
    assert(*w.nest.ip == 99);
    assert(*w.arr[0] == 9);
    assert(*w.arr[1] == 10);
    assert(p.x == 1);
  }
  return 0;
}
