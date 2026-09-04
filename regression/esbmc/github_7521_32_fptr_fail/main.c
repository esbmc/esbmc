#include <assert.h>
#include <stdint.h>

/* #7521: the pointer-to-integer cast is reachable only through a function
 * pointer. A reachability scan that stops at indirect calls misses it, widens
 * the pointer-free predicate over `struct wide`, loses the store through
 * `h.addr`, and turns this violation into VERIFICATION SUCCESSFUL. */

int a = 0, b = 0;
int nondet_int(void);

struct wide
{
  uintptr_t addr;
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

static void stash(struct wide *h)
{
  h->addr = nondet_int() ? (uintptr_t)&a : (uintptr_t)&b;
}

int main(void)
{
  struct wide h;
  void (*fp)(struct wide *) = stash;

  fp(&h);

  int *p = (int *)h.addr;
  *p = 5;

  assert(a == 0 && b == 0);
  return 0;
}
