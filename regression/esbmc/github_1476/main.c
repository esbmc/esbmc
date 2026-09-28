#include <assert.h>

/* Members named like the pads add_padding() mints, in structs/unions that do
   get padded. Before #1476 each of these produced two components of the same
   name, which member lookup cannot resolve. */

typedef struct
{
  int anon_pad$1;
  void *p;
} S;

typedef struct
{
  int a : 3;
  int anon_bit_field_pad$1;
} B;

typedef struct
{
  unsigned e : 7;
  int ext_int_pad$1;
} E;

typedef union
{
  char c[3];
  short s;
  short $pad;
} U;

int main()
{
  S s;
  s.anon_pad$1 = 42;
  assert(s.anon_pad$1 == 42);

  B b;
  b.anon_bit_field_pad$1 = 7;
  assert(b.anon_bit_field_pad$1 == 7);

  E e;
  e.ext_int_pad$1 = 9;
  assert(e.ext_int_pad$1 == 9);

  U u;
  u.$pad = 5;
  assert(u.s == 5);

  return 0;
}
