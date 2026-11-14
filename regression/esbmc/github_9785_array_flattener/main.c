/* #9785: with --array-flattener, which encodes arrays itself, the address
 * space stays one array of ranges; split in two it reported this pointer read
 * twice from one slot as two values. */
#include <stdlib.h>
int main(void) {
  void **slots = malloc(2 * sizeof(void *));
  if (!slots) return 0;
  void *first = slots[1];
  void *fresh = malloc(1);
  void *second = slots[1];
  __ESBMC_assert(first == second, "agree");
  return 0;
}
