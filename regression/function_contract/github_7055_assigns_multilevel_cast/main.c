/* The positive half of the cast case: stripping the cast must recover the root
 * without widening the frame, so a body writing only what the clause names
 * still verifies. */
typedef struct { int a; int b; } Inner;
typedef struct { void *sub; int x; } Outer;

void write_sub_a(Outer *o, int v)
{
  __ESBMC_requires(o != (void *)0 && o->sub != (void *)0);
  __ESBMC_assigns(((Inner *)o->sub)->a);
  __ESBMC_ensures(1);
  ((Inner *)o->sub)->a = v;
}

int main()
{
  return 0;
}
