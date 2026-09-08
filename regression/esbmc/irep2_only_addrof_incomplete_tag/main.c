/* `&x` on an object of incomplete struct or union type is the address of that
   object, not an array element: C11 6.5.3.2p3 gives it type "pointer to the
   operand's type". migrate_type lowers an incomplete tag to an infinitely
   sized uint8 array, so an adjuster that decays whatever ns.follow resolves to
   an array writes `&x[0]` here. */
struct incomplete;
union incomplete_u;

extern struct incomplete JJ;
extern union incomplete_u UU;

void take_struct(void *p);
void take_union(void *p);

int main(void)
{
  take_struct(&JJ);
  take_union(&UU);
  return 0;
}
