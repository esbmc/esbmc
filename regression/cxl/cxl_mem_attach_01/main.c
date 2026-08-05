// CXL memory device attach/detach lifecycle test.
// Tests that the CXL memory device lifecycle is correct:
// attach -> enable -> operations -> disable -> detach.
// Expected: VERIFICATION SUCCESSFUL

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <assert.h>

/* CXL device */
struct cxl_dev {
  void *regs;
  uint32_t dev_type;
};

/* CXL memory device */
struct cxl_mem {
  struct cxl_dev *cxld;
  bool attached;
  bool enabled;
  uint64_t dw8_size;
};

static struct cxl_dev test_cxld;
static struct cxl_mem test_cxlmem;

struct cxl_mem *cxl_mem_attach(struct cxl_dev *cxld)
{
  assert(cxld != NULL);
  test_cxlmem.cxld = cxld;
  test_cxlmem.attached = true;
  test_cxlmem.enabled = false;
  test_cxlmem.dw8_size = 4096;
  return &test_cxlmem;
}

void cxl_mem_detach(struct cxl_mem *cxlmem)
{
  assert(cxlmem != NULL);
  assert(cxlmem->attached == true);
  /* Must be disabled before detach */
  assert(cxlmem->enabled == false);
  cxlmem->attached = false;
  cxlmem->cxld = NULL;
}

int cxl_mem_enable(struct cxl_mem *cxlmem)
{
  assert(cxlmem != NULL);
  assert(cxlmem->attached == true);
  assert(cxlmem->enabled == false); /* Must not be already enabled */
  cxlmem->enabled = true;
  return 0;
}

void cxl_mem_disable(struct cxl_mem *cxlmem)
{
  assert(cxlmem != NULL);
  assert(cxlmem->attached == true);
  assert(cxlmem->enabled == true); /* Must be enabled to disable */
  cxlmem->enabled = false;
}

int main()
{
  test_cxld.regs = (void *)0x1000;
  test_cxld.dev_type = 1;

  /* Step 1: Attach */
  struct cxl_mem *cxlmem = cxl_mem_attach(&test_cxld);
  assert(cxlmem->attached == true);
  assert(cxlmem->enabled == false);

  /* Step 2: Enable */
  int ret = cxl_mem_enable(cxlmem);
  assert(ret == 0);
  assert(cxlmem->enabled == true);

  /* Step 3: Disable */
  cxl_mem_disable(cxlmem);
  assert(cxlmem->enabled == false);

  /* Step 4: Detach */
  cxl_mem_detach(cxlmem);
  assert(cxlmem->attached == false);
  assert(cxlmem->cxld == NULL);
}
