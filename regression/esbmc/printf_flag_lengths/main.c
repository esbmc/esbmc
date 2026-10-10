#include <assert.h>
#include <stdio.h>

char nondet_char(void);

int main(void)
{
  assert(printf("%+d", 5) == 2);
  assert(printf("% d", 5) == 2);
  assert(printf("%+d", -5) == 2);
  assert(printf("%#x", 255u) == 4);
  assert(printf("%#x", 0u) == 1);
  assert(printf("%#o", 8u) == 3);
  assert(printf("%.3d", 5) == 3);
  assert(printf("%.0d", 0) == 0);
  assert(printf("%5s", "ab") == 5);
  assert(printf("%.1s", "ab") == 1);

  char b[8];
  sprintf(b, "%+05d", 5);
  assert(b[0] == '+' && b[1] == '0' && b[4] == '5' && b[5] == '\0');
  sprintf(b, "%#X|%-2c|", 10u, 'q');
  assert(b[1] == 'X' && b[3] == '|' && b[5] == ' ' && b[7] == '\0');

  for (int i = 0; i < 7; i++)
    b[i] = nondet_char();
  b[7] = '\0';
  assert(printf("%.2s", b) <= 2);
  int unbounded = printf("%+.1f", 1.5);
  (void)unbounded;
  return 0;
}
