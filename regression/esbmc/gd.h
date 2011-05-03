#include "stubs.h"

/* Size of the input buffer. Since this example is a read overflow,
 * there is no output buffer. Must be at least 2 for things to work. */
#define INSZ BASE_SZ + 1

/* Size of a buffer used in gd_full.c; will affect a loop bound, so is
 * important for that example. */
#define ENTITY_NAME_LENGTH_MAX 8

/* The number of entities in entities[] and NR_OF_ENTITIES must
 * match. NR_OF_ENTITIES affects the number of iterations of search()
 * in gd_full_bad.c, so varying it should affect difficulty of that
 * example.
 *
 * Note that this is a *very* chopped-down array of entities -- see
 * entities.h in the gd sources for the real one. */
struct entities_s {
  char	*name;
  int	value;
};
struct entities_s entities[] = {
  {"AElig", 198},
  {"Aacute", 193},
  {"Acirc", 194},
};
#define NR_OF_ENTITIES 3

/* These things don't matter. */
#define Tcl_UniChar int
#define gdFTEX_Unicode 0
#define gdFTEX_Shift_JIS 1
#define gdFTEX_Big5 2

