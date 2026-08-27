/* value_sett::assign admits unions into its struct/union branch, then reaches
   is_subclass_of -- which is struct-only and casts both operands with
   to_struct_type. A union pair that is not base_type_eq therefore aborted with
   "to_struct_type() called on type whose type_id is union". Inheritance has no
   union analogue, so such a pair is simply incompatible. */
#include <assert.h>

union bits
{
  int : 5;
};

union payload
{
  int i;
  char c;
};

int main(void)
{
  union bits b = {};
  union payload p = {0};
  union payload q = p;
  assert(q.i == 0);
  (void)b;
  return 0;
}
