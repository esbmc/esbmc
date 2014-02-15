#include <stdlib.h>
#include <string.h>

typedef struct
{
  short a;
  unsigned short b;
  unsigned short c;
  unsigned long long Count;
  long long Count2;
} __attribute__((packed)) Struct1;
typedef struct
{
  short a;
  unsigned short b;
  unsigned short c;
  unsigned long long d;
  long long e;
  long long f;
} __attribute__((packed)) Struct2;
typedef union
{
  Struct1 a;
  Struct2 b;
} Union;
typedef struct
{
  int Count;
  Union List[1];
} __attribute__((packed)) Struct3;
unsigned long long Sum (Struct3 *instrs) __attribute__((noinline));
unsigned long long Sum (Struct3 *instrs)
{
    unsigned long long count = 0;
    int i;
    for (i = 0; i < instrs->Count; i++) {
        count += instrs->List[i].a.Count;
    }
    return count;
}
long long Sum2 (Struct3 *instrs) __attribute__((noinline));
long long Sum2 (Struct3 *instrs)
{
    long long count = 0;
    int i;
    for (i = 0; i < instrs->Count; i++) {
        count += instrs->List[i].a.Count2;
    }
    return count;
}
static void dummy_abort(void)
{
}
int main() {
  Struct3 *p = malloc (sizeof (int) + 3 * sizeof(Union));
__ESBMC_assume(p != ((void *)0));
//  memset(p, 0, sizeof(int) + 3*sizeof(Union));
  p->Count = 3; // XXX Previously this assignment would be knackered by the
                // concat interpretation code, and assign 3000303 instead.
  p->List[0].a.Count = 555;
  p->List[1].a.Count = 999;
  p->List[2].a.Count = 0x101010101ULL;
  p->List[0].a.Count2 = 555;
  p->List[1].a.Count2 = 999;
  p->List[2].a.Count2 = 0x101010101LL;
  if (Sum(p) != 555 + 999 + 0x101010101ULL)
    dummy_abort();
  if (Sum2(p) != 555 + 999 + 0x101010101LL)
    dummy_abort();
  p = ((void *)0);
  return 0;
}
