// esbmc/esbmc#4715 (V.4.4 parity): negative variant of label_decl_01. With the
// labeled decl-with-init kept live across the --irep2-bodies round-trip, `s`
// genuinely points at `p`, so the wrong-value assertion below is a real
// violation (s[0] is 'h', not 'x'). Guards against the positive test passing
// vacuously (e.g. if `s` were dead and read an unconstrained value that could
// also satisfy the assert).
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
  assert(t.get() == 'x'); // wrong: t.get() == 'h'
  return 0;
}
