/* The fold must leave alone any pointer whose object is not fixed: a
 * pointer chosen between two objects may equal either, and a pointer from
 * a dereference may alias anything, including another object's address. */
struct S
{
  int x;
};
struct S s;
int a[4], b[4];
int *q;
int nondet_int(void);

int main(void)
{
  int *p = nondet_int() ? &a[0] : &b[0];
  q = nondet_int() ? &a[2] : &b[2];
  __ESBMC_assert(p != &a[0], "a chosen pointer may equal either object");
  __ESBMC_assert(&q[-2] != &b[0], "a dereference-rooted address may too");
  struct S *r = &s;
  __ESBMC_assert(&r->x != &s.x, "a member through a pointer may too");
  return 0;
}
