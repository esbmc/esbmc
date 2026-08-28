#include <stdint.h>
#include <stddef.h>
#include <assert.h>

/* Companion negative case to github_7311: the last buffered byte was written
 * through the struct-field-derived index c->buf[c->buflen] and is 0, so the
 * assertion below must fail. Guards the index-folding optimization against
 * dropping reachable violations. */

typedef struct
{
  uint8_t buf[64];
  size_t buflen;
  uint64_t len;
} ctx;

static void update(ctx *c, const uint8_t *p, size_t n)
{
  c->len += n;
  if (c->buflen)
  {
    size_t need = 64 - c->buflen;
    if (n < need)
    {
      c->buf[c->buflen] = *p;
      c->buflen += n;
      return;
    }
    c->buf[c->buflen] = *p;
    c->buflen = 0;
    p += need;
    n -= need;
  }
  while (n >= 64)
  {
    p += 64;
    n -= 64;
  }
  if (n)
  {
    c->buf[0] = *p;
    c->buflen = n;
  }
}

int main(void)
{
  ctx c;
  c.buflen = 0;
  c.len = 0;
  uint8_t b = 0x80;
  update(&c, &b, 1);
  b = 0;
  while (c.buflen != 2)
    update(&c, &b, 1);
  /* c.buflen == 2 here and c.buf[1] == 0 (written via c->buf[c->buflen]). */
  assert(c.buf[c.buflen - 1] != 0);
  return 0;
}
