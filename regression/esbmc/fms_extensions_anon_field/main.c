#include <assert.h>
#include <stddef.h>

/* An unnamed field of a named struct type is a Microsoft/plan9 extension used
 * by the Linux kernel (e.g. struct filename in include/linux/fs.h). It only
 * parses under -fms-extensions, and the embedded struct must contribute its
 * full size and allow promoted member access. */
struct head
{
  const char *name;
  int refcnt;
};

struct filename
{
  struct head;
  char iname[8];
};

_Static_assert(sizeof(struct head) == 16, "head layout");
_Static_assert(offsetof(struct filename, iname) == 16, "promoted field offset");
_Static_assert(sizeof(struct filename) == 24, "embedded struct contributes size");

int main(void)
{
  struct filename f;
  f.refcnt = 5;
  f.iname[0] = 'x';
  assert(f.refcnt == 5);
  assert(f.iname[0] == 'x');
  return 0;
}
