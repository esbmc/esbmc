#include <assert.h>
#include <stdbool.h>

/* CPROVER type ids used as ordinary C identifiers. An irep node for a name
   carries the identifier text as its id(), so a type rewrite that keys on
   id() alone corrupts these -- a function named c_bool had its name rewritten
   to `signedbv`, and the program then verified to FAILED. */
int c_bool(void)
{
  return 5;
}

int c_enum_tag(void)
{
  return 6;
}

/* The genuine types must still be rewritten. */
enum Colour
{
  RED,
  GREEN,
  BLUE
};

struct S
{
  _Bool flag;
  enum Colour c;
  unsigned int bits : 3;
};

int main(void)
{
  struct S s;
  s.flag = true;
  s.c = GREEN;
  s.bits = 5;
  assert(c_bool() == 5);
  assert(c_enum_tag() == 6);
  assert(s.flag);
  assert(s.c == 1);
  assert(s.bits == 5);
  return 0;
}
