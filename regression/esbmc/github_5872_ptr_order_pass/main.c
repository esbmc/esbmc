#include <stddef.h>

/* With the same-object check disabled, relational comparison of pointers to
 * distinct objects must still behave as a consistent total order: within one
 * object it is the C-defined offset order, across objects the direction is
 * arbitrary but the algebraic properties of the operator must hold. */
int main()
{
  char a, b;
  char *p1 = &a, *p2 = &b;
  int le = (p1 <= p2);
  int ge = (p2 <= p1);
  /* antisymmetry: for distinct objects at most one direction holds */
  __ESBMC_assert(!(le && ge), "antisymmetry");
  /* totality: at least one direction must hold */
  __ESBMC_assert(le || ge, "totality");
  /* strict order consistency */
  int lt = (p1 < p2), gt = (p1 > p2);
  __ESBMC_assert(!(lt && gt), "lt/gt exclusive");
  __ESBMC_assert(lt || gt, "trichotomy for distinct objects");
  /* same-object comparisons keep the C-defined offset order */
  char arr[4];
  __ESBMC_assert(&arr[1] < &arr[3], "in-object order");
  __ESBMC_assert(!(&arr[3] <= &arr[1]), "in-object strictness");
  return 0;
}
