/* Distilled from symbolically executing a JavaCard VM's test fixture:
 * a CAP-file parse tree emitted as a static array of structs whose
 * child pointers point back into the same array, walked by pointer
 * arithmetic. Each step's offset must fold, or every field read
 * becomes a symbolic-offset byte extract over the whole aggregate and
 * the solver diverges. */
#include <string.h>
typedef struct node
{
  const char *name;
  struct node *kids;
  int nkids;
  int val;
} node;
static node t[7] = {
  {"root", (node *)(t + 1), 2, 0},
  {"alpha", (node *)(t + 3), 2, 1},
  {"beta", (node *)(t + 5), 2, 2},
  {"gamma", 0, 0, 3},
  {"delta", 0, 0, 4},
  {"epsilon", 0, 0, 5},
  {"zeta", 0, 0, 6},
};
static const node *child(const node *n, const char *nm)
{
  for (int i = 0; i < n->nkids; i++)
    if (n->kids[i].name && strcmp(n->kids[i].name, nm) == 0)
      return &n->kids[i];
  return 0;
}
int main(void)
{
  const node *a = child(t, "alpha");
  const node *d = child(a, "delta");
  const node *g = d - 1; /* negative-constant step, also the add arm */
  __ESBMC_assert(
    d != 0 && d->val == 4 && g->val == 3,
    "struct-array pointer arithmetic folds");
  return 0;
}
