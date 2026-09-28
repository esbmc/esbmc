/* Three levels, so the search for the parameter the path is rooted at has to
 * recurse rather than look one step down. */
typedef struct { int deep; } L3;
typedef struct { L3 *l3; int mid; } L2;
typedef struct { L2 *l2; int top; } L1;

void f(L1 *p, int v)
{
  __ESBMC_requires(
    p != (void *)0 && p->l2 != (void *)0 && p->l2->l3 != (void *)0);
  __ESBMC_assigns(p->l2->l3->deep);
  __ESBMC_ensures(1);
  p->l2->l3->deep = v;
  p->top = 99; /* not in __ESBMC_assigns */
}

int main()
{
  return 0;
}
