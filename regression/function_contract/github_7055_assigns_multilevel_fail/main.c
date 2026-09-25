/* A path through more than one pointer used to reach neither check: it was not
 * a `ptr->field` the per-field machinery recognises, and recording it as a
 * direct target matched no snapshot, so the function generated ZERO VCCs and a
 * body writing outside its frame verified. */
typedef struct { int a; int b; } Inner;
typedef struct { Inner *sub; int x; } Outer;

void write_sub_a(Outer *o, int v)
{
  __ESBMC_requires(o != (void *)0 && o->sub != (void *)0);
  __ESBMC_assigns(o->sub->a);
  __ESBMC_ensures(1);
  o->sub->a = v;
  o->x = 99; /* not in __ESBMC_assigns: a frame violation */
}

int main()
{
  return 0;
}
