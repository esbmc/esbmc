#include <assert.h>
#include <stdlib.h>
#include <string.h>

struct S {
  int x;
  int y[];
};

void dynamic_fam_test_1(int x) {
  // Allocate FAM
  struct S *s = (struct S*) malloc(sizeof(int) + sizeof(int) * x);

  // This is probably checked by clang itself!
  assert(sizeof(s) == 8);
  assert(sizeof(s->x) == 4);
  assert(sizeof(*s) == 4);

  s->x = x;

  // Initialize FAM with i
  for(int i = 0; i < s->x; i++)
    s->y[i] = i;

  // Check if FAM was actually initialized (Check #1)
  for(int i = 0; i < s->x; i++)
    assert(s->y[i] == i);

  free(s);
}

void dynamic_fam_test_2()
{
  int y[] = {0, 1, 2, 3};
  int x = sizeof(y)/sizeof(y[0]);
  assert(x == 4);

  // Allocate FAM
  struct S *s = (struct S*) malloc(sizeof(int) * (x + 1));

  // Initialize it with Y

  // Wrong test
  memcpy(s->y, y, sizeof(s->y[0]) * x);

  // Check
  for(int i = 0; i < x; i++)
    assert(s->y[i] == i);

  memset(s->y, 0, sizeof(s->y[0]) * x);
  for(int i = 0; i < x; i++)
    assert(!s->y[i]);

  free(s);
}

int main() {
  dynamic_fam_test_1(3);
  dynamic_fam_test_2();
}
