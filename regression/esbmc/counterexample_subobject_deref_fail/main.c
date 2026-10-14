// The dereference resolves to a sub-object, so the SSA assignment names the
// enclosing `o` while the lvalue names a member of `o.in`. Reporting that
// member against `o` would find `outer::g` and print 9 for a write of 5, so
// the trace must report the whole object here instead.
struct inner
{
  int pad;
  int g;
};

struct outer
{
  int g;
  struct inner in;
};

struct outer o;

static void set(struct inner *p)
{
  p->g = 5;
}

int main(void)
{
  o.g = 9;
  set(&o.in);
  __ESBMC_assert(o.in.g != 5, "in.g is 5");
  return 0;
}
