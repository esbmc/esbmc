#include "stubs.h"

/* Make u_char be a char. If we use unsigned chars, SatAbs gives us an
 * error whenever we use strlen, since it won't iterate over a string
 * of unsigned chars passed as chars.
 */
typedef char u_char;
typedef int u_int;
typedef int u_int32_t;

/* Buffer being overflowed has size (MAXDATA*2); I believe this is
 * because its a buffer of bytes, and two bytes keep being written at
 * a time.
 * 
 * Overflowed buffers in rrextract-sig/ may have an additional
 * SPACE_FOR_VARS elements. */
#define MAXDATA BASE_SZ

/* Input buffer has this size, plus some constant SPACE_FOR_VARS dependent on how many
 * bytes get skipped before the operations involved in the
 * overflow. This constant is different in different variants.
 */
#define MSGLEN  MAXDATA + 2

/* We don't loop over this, so we don't really care what it is. */
#define NAMELEN 3

#define INT16SZ 2
#define INT32SZ 4

#define CLASS_MAX 100
#define MAXIMUM_TTL 101

/* Macros rrextract() uses */
#define GETSHORT(to, from) \
  do {(to) = nondet_short(); (from) += INT16SZ;} while(0)
#define GETLONG(to, from) \
  do {(to) = nondet_long(); (from) += INT32SZ;} while(0)
#define BOUNDS_CHECK(ptr, count) \
  do {if ((ptr) + (count) > eom) return -1;} while(0)

/* dn_expand -- "domain name expand"
 *   -- expands comp_dn (compressed domain name) to exp_dn (full domain name)
 *   -- returns -1 on error, or else strlen(comp_dn)
 */
int dn_expand(const u_char *msg, const u_char *eomorig,
              const u_char *comp_dn, char *exp_dn, int length);


extern int nondet_int();

extern int nondet_short();

extern int nondet_short();
