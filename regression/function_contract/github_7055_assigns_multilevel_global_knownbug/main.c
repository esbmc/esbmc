/* Rooting the path recovers the root symbol, but only a *parameter* gets a
 * snapshot: materialize_ptr_field_snapshots walks the function's arguments, so
 * a path rooted at a global pointer records a field target that matches
 * nothing and the function still generates ZERO VCCs. */
typedef struct { int a; int b; } Inner;
typedef struct { Inner *sub; int x; } Outer;

Outer *g;

void write_sub_a(int v)
{
  __ESBMC_requires(g != (void *)0 && g->sub != (void *)0);
  __ESBMC_assigns(g->sub->a);
  __ESBMC_ensures(1);
  g->sub->a = v;
  g->x = 99; /* not in __ESBMC_assigns */
}

int main()
{
  return 0;
}
