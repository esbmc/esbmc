// CXL HDM decoder setup test.
// Tests that HDM decoders are set up without overlapping regions.
// Expected: VERIFICATION SUCCESSFUL

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

int cxl_setup_hdm_decoders(struct cxl_dev *cxld,
                           uint64_t base, uint64_t size, int decoder_idx)
{
  assert(cxld != NULL);
  assert(decoder_idx >= 0 && decoder_idx < CXL_HDM_DECODER_MAX);
  assert(size > 0);

  uint64_t limit = base + size - 1;

  /* Check for overlap with existing enabled decoders */
  for (int i = 0; i < cxld->num_decoders; i++)
  {
    if (!cxld->decoders[i].enabled)
      continue;
    /* Check overlap: two ranges [a1,b1] and [a2,b2] overlap if a1<=b2 && a2<=b1 */
    if (base <= cxld->decoders[i].limit &&
        cxld->decoders[i].base <= limit)
    {
      return -1; /* Overlap */
    }
  }

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
  assert(test_cxld.decoders[0].enabled == 1);
  assert(test_cxld.decoders[0].base == 0);
  assert(test_cxld.decoders[0].limit == 256 * 1024 * 1024 - 1);

  /* Setup decoder 1: 256MB-512MB (adjacent, no overlap) */
  ret = cxl_setup_hdm_decoders(&test_cxld, 256 * 1024 * 1024,
                                256 * 1024 * 1024, 1);
  assert(ret == 0);
  assert(test_cxld.decoders[1].enabled == 1);

  /* Setup decoder 2: 512MB-768MB */
  ret = cxl_setup_hdm_decoders(&test_cxld, 512 * 1024 * 1024,
                                256 * 1024 * 1024, 2);
  assert(ret == 0);

  /* Verify no overlap between decoder 0 and 2 */
  assert(test_cxld.decoders[0].limit < test_cxld.decoders[2].base);
}
