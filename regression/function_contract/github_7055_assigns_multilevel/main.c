/* The positive half: a body that writes only what the clause names must still
 * verify, so rooting the path at its parameter does not over-report. */
typedef struct { int a; int b; } Inner;
typedef struct { Inner *sub; int x; } Outer;

void write_sub_a(Outer *o, int v)
{
  __ESBMC_requires(o != (void *)0 && o->sub != (void *)0);
  __ESBMC_assigns(o->sub->a);
  __ESBMC_ensures(1);
  o->sub->a = v;
}

int main()
{
  return 0;
}
