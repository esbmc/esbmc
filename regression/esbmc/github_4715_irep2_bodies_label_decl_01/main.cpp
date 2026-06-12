// esbmc/esbmc#4715 (V.4.4 parity): a labeled declaration with initializer
// `lbl: char *s = p;` is legacy label(decl-block(decl)). Under --irep2-bodies
// the decl-block round-tripped to a code("block"), whose scope boundary made
// convert_block emit a premature DEAD for `s` right after its init -- so every
// later use read a dead object. This is the pattern at the top of the C++
// std::string operational-model methods (`__ESBMC_HIDE: char *s = ...;`), which
// produced wrong string contents under the flag. With the fix the single-decl
// labeled decl-block round-trips as label(decl) and the DEAD is deferred to the
// enclosing scope, so `s` stays live and the read is correct.
#include <cassert>

struct T
{
  char *p;
  char get()
  {
  lbl:
    char *s = p;
    return s[0];
  }
};

int main()
{
  char buf[2] = {'h', 0};
  T t;
  t.p = buf;
  assert(t.get() == 'h');
  return 0;
}
