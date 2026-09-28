/* The entry-value scan walks back at most kMaxEntryScanBack instructions from
 * the loop head. Here the counter's initialisation sits further back than that
 * behind a long straight-line prologue, so no entry value is established and
 * the recogniser declines rather than summarise against an unknown i0. */
#include <assert.h>

int main(void)
{
  unsigned int n;
  __ESBMC_assume(n <= 3);

  unsigned int i = 0;
  unsigned int s = 0;
  unsigned int d = 0;

  d = 1;
  d = 2;
  d = 3;
  d = 4;
  d = 5;
  d = 6;
  d = 7;
  d = 8;
  d = 9;
  d = 10;
  d = 11;
  d = 12;
  d = 13;
  d = 14;
  d = 15;
  d = 16;
  d = 17;
  d = 18;
  d = 19;
  d = 20;
  d = 21;
  d = 22;
  d = 23;
  d = 24;
  d = 25;
  d = 26;
  d = 27;
  d = 28;
  d = 29;
  d = 30;
  d = 31;
  d = 32;
  d = 33;
  d = 34;
  d = 35;
  d = 36;
  d = 37;
  d = 38;
  d = 39;
  d = 40;
  d = 41;
  d = 42;
  d = 43;
  d = 44;
  d = 45;
  d = 46;
  d = 47;
  d = 48;
  d = 49;
  d = 50;
  d = 51;
  d = 52;
  d = 53;
  d = 54;
  d = 55;
  d = 56;
  d = 57;
  d = 58;
  d = 59;
  d = 60;
  d = 61;
  d = 62;
  d = 63;
  d = 64;
  d = 65;
  d = 66;
  d = 67;
  d = 68;
  d = 69;
  d = 70;

  while (i < n)
  {
    s = s + 1;
    i = i + 1;
  }

  assert(s == n);
  return 0;
}
