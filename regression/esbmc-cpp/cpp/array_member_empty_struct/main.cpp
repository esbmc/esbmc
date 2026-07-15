// Regression for goto-symex: assigning a re-constituted array literal.
//
// A struct whose only member is an array of an empty class type produces,
// after construction, a value whose array member is a constant_array. When
// that value is written back (e.g. `c = tmp$1`), symex_assign_structure
// projects the array member out as a constant_array *lhs*. symex_assign_rec
// had no case for constant_array and aborted with
// "assignment to constant_array not handled". It now projects each element,
// mirroring the constant_struct / constant_union handling.
struct E
{
};

struct C
{
  E a[3];
};

int main()
{
  C c;
  (void)c;
  return 0;
}
