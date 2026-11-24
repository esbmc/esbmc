#include <stdint.h>

#define WORDS 4u
#define PAGES 3u

typedef struct
{
  uint32_t cfg_count;
  uint8_t mode;
  uint64_t timestamp;
  uint32_t hdr[PAGES * WORDS];
  uint32_t table[4];
} state_t;

static state_t state;
unsigned nondet_uint(void);

static void dma_read(uint32_t *dest, unsigned bytes)
{
  uint32_t tmp[WORDS];
  __ESBMC_init_object(tmp);
  __ESBMC_unroll(WORDS);
  for (unsigned i = 0; i < bytes / sizeof(uint32_t); ++i)
    dest[i] = tmp[i];
}

static void load_header(state_t *s, unsigned bytes)
{
  for (unsigned page = 0; page < PAGES; ++page)
  {
    unsigned chunk = bytes > WORDS * 4u ? WORDS * 4u : bytes;
    dma_read(&s->hdr[page * WORDS], chunk);
    bytes -= chunk;
  }
}

static void save_once(state_t *s)
{
  unsigned size = nondet_uint();
  __ESBMC_assume(size <= PAGES * WORDS * 4u);
  load_header(s, size);
  s->table[s->cfg_count] = s->hdr[0];
}

int main(void)
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n <= 1000000u);
  __ESBMC_init_object(&state);
  __ESBMC_assume(state.cfg_count <= 4);
  for (unsigned i = 0; i < n; ++i)
    save_once(&state);
}
