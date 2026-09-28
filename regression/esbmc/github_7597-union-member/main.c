/* The written member is a union, so the value pinning re-offers is union-typed
 * and was refused: unions were held out of the carried-read class because their
 * members alias. A *read* aliases nothing extra -- member2t::do_simplify steps
 * past a `with` whose source is a union at no member, and
 * fold_union_member_read projects out of a constant_union2t only -- so the
 * counter beside it folds while a cross-member access stays a real punning
 * read. Same-width members, so the punning assertion does not depend on the
 * target's endianness. No --unwind: the bound has to come from propagation. */
#include <assert.h>

union mix
{
  unsigned int word;
  int as_signed;
};

struct plc
{
  int step;
  union mix m;
};

struct io
{
  unsigned int in;
};

struct plc VAR;
struct io IO;

unsigned int nondet_uint(void);

int main(void)
{
  IO.in = nondet_uint();
  /* A merge point, so IO carries no propagated constant afterwards. */
  if (IO.in > 1000u)
    IO.in = 1000u;

  for (VAR.step = 1; VAR.step <= 5; VAR.step = VAR.step + 1)
    VAR.m.word = IO.in + 1u;
  assert(VAR.step == 6);
  assert(VAR.m.as_signed == (int)(IO.in + 1u));

  for (VAR.step = 1; VAR.step <= 6; VAR.step = VAR.step + 1)
    if (IO.in > 0u)
      VAR.m.word = IO.in + 2u;
  assert(VAR.step == 7);

  return 0;
}
