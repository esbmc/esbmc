// CXL HDM decoder overlap detection test.
// Tests that the driver correctly rejects overlapping HDM decoder
// configurations.
// Expected: VERIFICATION FAILED (driver bug: allows overlapping decoders)

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

#define CXL_HDM_DECODER_MAX 8

struct cxl_hdm_decoder {
  uint64_t base;
  uint64_t limit;
  int enabled;
};

struct cxl_dev {
  struct cxl_hdm_decoder decoders[CXL_HDM_DECODER_MAX];
  int num_decoders;
};

static struct cxl_dev test_cxld;

/*
 * BUG: This driver does NOT check for overlap between decoders.
 * A correct implementation must verify that new decoder ranges
 * don't overlap with existing enabled decoders.
 */
int cxl_setup_hdm_decoders(struct cxl_dev *cxld,
                           uint64_t base, uint64_t size, int decoder_idx)
{
  assert(cxld != NULL);
  assert(decoder_idx >= 0 && decoder_idx < CXL_HDM_DECODER_MAX);
  assert(size > 0);

  uint64_t limit = base + size - 1;

  /* BUG: No overlap check! */

  cxld->decoders[decoder_idx].base = base;
  cxld->decoders[decoder_idx].limit = limit;
  cxld->decoders[decoder_idx].enabled = 1;
  if (decoder_idx >= cxld->num_decoders)
    cxld->num_decoders = decoder_idx + 1;

  return 0;
}

int main()
{
  test_cxld.num_decoders = 0;
  for (int i = 0; i < CXL_HDM_DECODER_MAX; i++)
    test_cxld.decoders[i].enabled = 0;

  /* Setup decoder 0: 0-256MB */
  int ret = cxl_setup_hdm_decoders(&test_cxld, 0, 256 * 1024 * 1024, 0);
  assert(ret == 0);

  /*
   * BUG: Setup decoder 1 overlapping with decoder 0 (128MB-384MB).
   * The driver should reject this but doesn't.
   */
  ret = cxl_setup_hdm_decoders(&test_cxld, 128 * 1024 * 1024,
                                256 * 1024 * 1024, 1);
  assert(ret == 0); /* Bug: returns success for overlapping range */

  /*
   * The invariant: no two enabled decoders should overlap.
   * Two ranges [a1,b1] and [a2,b2] overlap if a1<=b2 && a2<=b1.
   */
  for (int i = 0; i < test_cxld.num_decoders; i++)
  {
    if (!test_cxld.decoders[i].enabled)
      continue;
    for (int j = i + 1; j < test_cxld.num_decoders; j++)
    {
      if (!test_cxld.decoders[j].enabled)
        continue;

      /* Check overlap */
      uint64_t a1 = test_cxld.decoders[i].base;
      uint64_t b1 = test_cxld.decoders[i].limit;
      uint64_t a2 = test_cxld.decoders[j].base;
      uint64_t b2 = test_cxld.decoders[j].limit;

      __ESBMC_assert(!(a1 <= b2 && a2 <= b1),
                     "HDM decoders overlap");
    }
  }
}
