#include <assert.h>

struct Pair
{
  unsigned value;
  unsigned tag;
};

struct Box
{
  struct Pair pair;
  unsigned count;
};

extern struct Box nondet_box(void);
extern unsigned nondet_uint(void);

int main(void)
{
  struct Box input = nondet_box();
  struct Box slots[2] = {0};
  unsigned index = nondet_uint();
  __ESBMC_assume(index < 2);
  slots[index] = input;
  slots[index].pair.tag = 7;
  assert(slots[index].pair.value == input.pair.value);
  assert(slots[index].count == input.count);
  assert(slots[1 - index].pair.value == 0);
  assert(slots[index].pair.tag == 7);
  return 0;
}
