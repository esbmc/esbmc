/* A union whose own tag appears under a pointer inside its body, and a
   mutually-referential pair. No other test in the suite has either, and the
   union tag is the one shape get_struct_union_class's completeness check does
   not name -- it compares against "incomplete_struct" only (esbmc/esbmc#4715).
   The assertions read a member other than the one written, so they hold only
   if the tag really is a union. */
#include <assert.h>

union node { int a; int b; union node *next; };

union fwd;
union holder { int x; int y; union fwd *f; };
union fwd { union holder *o; int w; };

int main(void)
{
  union node n;
  n.next = 0;
  n.a = 5;

  union holder h;
  h.f = 0;
  h.x = 7;

  union fwd ff;
  ff.o = &h;

  assert(n.b == 5);
  assert(h.y == 8);
  return 0;
}
