#include <stdint.h>
#include <stddef.h>

/* Minimized from a streaming SHA-1 implementation: a callee whose body has a
 * buffer-flush loop after an early-return buffer path, called in a loop.
 * Dereferencing c->buf[c->buflen] used to produce a whole-struct byte_update
 * because the index was symbolic at dereference time, dropping the struct's
 * recorded constant; every guard over c.buflen then became undecidable and
 * both loops unwound to the bound (quadratic symex time). */

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
  /* Runs exactly one iteration; only verifiable quickly if the write above
   * keeps c.buflen constant-propagated. */
  while (c.buflen != 2)
    update(&c, &b, 1);
  return c.buf[0];
}
