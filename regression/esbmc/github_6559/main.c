/* Distilled from MATIEC's accessor.h idiom (issue #6559): __INIT_VAR leaves
 * the flags byte indeterminate (flags |= 0), and __SET_VAR skips its write
 * when the FORCE bit is set. Zeroing the instance removes the
 * nondeterminism, so the guarded write must take effect. */
#include <assert.h>
#include <string.h>

#define FORCE_FLAG 0x02
#define RETAIN_FLAG 0x04

typedef struct
{
  unsigned char value;
  unsigned char flags;
} iec_var;

typedef struct
{
  iec_var Q;
  iec_var PT;
} fb_data;

#define SET_VAR(d, name, v)                                                    \
  if(!((d).name.flags & FORCE_FLAG))                                           \
  (d).name.value = v

static void fb_init(fb_data *d, int retain)
{
  d->Q.value = 0;
  d->Q.flags |= retain ? RETAIN_FLAG : 0;
  d->PT.value = 0;
  d->PT.flags |= retain ? RETAIN_FLAG : 0;
}

int main(void)
{
  fb_data t;
  memset(&t, 0, sizeof(t));
  fb_init(&t, 0);
  SET_VAR(t, PT, 100);
  assert(t.PT.value == 100);
  return 0;
}
