#include <string.h>

/* A member-less union occupies no storage, so there is no member for the
   byte-wise memset to write into: ESBMC has to fall back on __memset_impl.
   Reaching that fallback needs --no-pointer-check, which drops both the
   out-of-bounds claim that would otherwise cut the path short and the empty
   union dereference failure __memset_impl itself raises. */
union empty
{
};

int main()
{
  union empty u;
  memset(&u, 0, 4);
  return 0;
}
