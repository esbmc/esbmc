#include <assert.h>

/* `string` is an ordinary C identifier. A CPROVER irep node for a name holds
   the identifier text as its id(), so a type-blind check for the `string` type
   would decline all three of these spellings. */
struct S
{
  int string;
};

int len(const char *string)
{
  return string ? 1 : 0;
}

int string(void)
{
  return 7;
}

int main(void)
{
  struct S v;
  v.string = 3;
  assert(v.string == 3);
  assert(len("x") == 1);
  assert(string() == 7);
  return 0;
}
