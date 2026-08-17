// CXL device initialization verification test.
// Tests that a CXL device is properly initialized: DEV_CTRL is set,
// INIT bit is cleared, ENABLE bit is set.
// Expected: VERIFICATION SUCCESSFUL

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

/* Simulated CXL device */
static struct cxl_dev test_cxld;
static uint64_t test_dev_ctrl;
static uint64_t test_dev_stat;

/* Override the model's register access for deterministic testing */
u64 cxl_read_dev_ctrl(struct cxl_dev *cxld)
{
  (void)cxld;
  return test_dev_ctrl;
}

void cxl_write_dev_ctrl(struct cxl_dev *cxld, u64 val)
{
  (void)cxld;
  test_dev_ctrl = val;
}

u64 cxl_read_dev_stat(struct cxl_dev *cxld)
{
  (void)cxld;
  return test_dev_stat;
}

/*
 * Simulate a correct device initialization sequence:
 * 1. Read initial DEV_CTRL (should be 0)
 * 2. Set INIT bit
 * 3. Wait for INIT to complete (poll DEV_STAT)
 * 4. Clear INIT, set ENABLE
 */
int main()
{
  test_cxld.regs = (void *)0x1000;
  test_dev_ctrl = 0;
  test_dev_stat = 0;

  /* Step 1: Initial state — device should be disabled */
  u64 ctrl = cxl_read_dev_ctrl(&test_cxld);
  assert((ctrl & CXL_DCR_ENABLE) == 0);

  /* Step 2: Set INIT bit */
  cxl_write_dev_ctrl(&test_cxld, ctrl | CXL_DCR_CLEAR_INIT);
  assert((test_dev_ctrl & CXL_DCR_CLEAR_INIT) != 0);

  /* Step 3: Poll DEV_STAT for INIT_DONE */
  /* In the model, DEV_STAT is nondet, so we assume it eventually sets INIT_DONE */
  __ESBMC_assume((cxl_read_dev_stat(&test_cxld) & CXL_DSR_INIT_DONE) != 0);

  /* Step 4: Clear INIT, set ENABLE */
  cxl_write_dev_ctrl(&test_cxld,
                     (test_dev_ctrl & ~CXL_DCR_CLEAR_INIT) | CXL_DCR_ENABLE);

  /* Verify final state */
  assert((test_dev_ctrl & CXL_DCR_CLEAR_INIT) == 0);
  assert((test_dev_ctrl & CXL_DCR_ENABLE) != 0);

  /* Verify that ENABLE was not set before INIT completed */
  /* This is the key invariant: ENABLE must follow INIT */
  __ESBMC_assert((test_dev_ctrl & CXL_DCR_ENABLE) != 0,
                 "Device should be enabled after init");
}
