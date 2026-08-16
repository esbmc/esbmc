// github #5868 gap 5: the intrinsic prelude was appended to clang's predefines
// buffer *after* the `#include` lines that --include-file generates, so a forced
// header reaching ESBMC's own models failed with `use of undeclared identifier
// 'nondet_uint'`. main.cpp itself needs nothing special -- the whole point is
// that the forced header is processed before the intrinsics are in scope.
int main()
{
  assert(helper() == 3);
  return 0;
}
