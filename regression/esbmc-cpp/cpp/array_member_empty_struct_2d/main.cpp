// Nested-array variant of array_member_empty_struct: the array member is
// multidimensional, so projecting the constant_array lhs recurses (the inner
// array element is itself a constant_array). Without the recursive handling
// this aborts with "assignment to constant_array not handled".
struct E
{
};

struct C
{
  E a[2][2];
};

int main()
{
  C c;
  (void)c;
  return 0;
}
