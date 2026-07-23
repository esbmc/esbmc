// CXL memory partition state machine test.
// Tests that partition state transitions follow the CXL spec.
// Expected: VERIFICATION SUCCESSFUL

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

/* CXL partition states per CXL 2.0 spec */
enum cxl_partition_state {
  CXL_PARTITION_STATE_UNPARTITIONED = 0,
  CXL_PARTITION_STATE_SPLIT_DATA_PMEM = 1,
  CXL_PARTITION_STATE_NONVOL_DATA_PMEM = 2,
  CXL_PARTITION_STATE_VOLATILE_DATA_PMEM = 3,
};

struct cxl_mem {
  enum cxl_partition_state current_state;
  uint64_t total_size;
  uint64_t data_size;
  uint64_t pmem_size;
};

static struct cxl_mem test_cxlmem;

int cxl_mem_set_partition_state(struct cxl_mem *cxlmem,
                                uint64_t split_data_size,
                                uint64_t split_pmem_size)
{
  assert(cxlmem != NULL);
  assert(split_data_size + split_pmem_size <= cxlmem->total_size);

  /* Valid transition: UNPARTITIONED -> SPLIT */
  if (cxlmem->current_state == CXL_PARTITION_STATE_UNPARTITIONED)
  {
    cxlmem->data_size = split_data_size;
    cxlmem->pmem_size = split_pmem_size;
    cxlmem->current_state = CXL_PARTITION_STATE_SPLIT_DATA_PMEM;
    return 0;
  }

  /* Only allow transitions from SPLIT state */
  if (cxlmem->current_state != CXL_PARTITION_STATE_SPLIT_DATA_PMEM)
    return -1;

  cxlmem->data_size = split_data_size;
  cxlmem->pmem_size = split_pmem_size;
  return 0;
}

int main()
{
  test_cxlmem.total_size = 1024 * 1024 * 1024; /* 1 GB */
  test_cxlmem.current_state = CXL_PARTITION_STATE_UNPARTITIONED;
  test_cxlmem.data_size = 0;
  test_cxlmem.pmem_size = 0;

  /* Transition: UNPARTITIONED -> SPLIT (512MB data, 512MB pmem) */
  int ret = cxl_mem_set_partition_state(&test_cxlmem,
                                         512 * 1024 * 1024,
                                         512 * 1024 * 1024);
  assert(ret == 0);
  assert(test_cxlmem.current_state == CXL_PARTITION_STATE_SPLIT_DATA_PMEM);
  assert(test_cxlmem.data_size == 512 * 1024 * 1024);
  assert(test_cxlmem.pmem_size == 512 * 1024 * 1024);

  /* Verify total size is preserved */
  assert(test_cxlmem.data_size + test_cxlmem.pmem_size == test_cxlmem.total_size);

  /* Transition: SPLIT -> SPLIT (new split) */
  ret = cxl_mem_set_partition_state(&test_cxlmem,
                                     256 * 1024 * 1024,
                                     768 * 1024 * 1024);
  assert(ret == 0);
  assert(test_cxlmem.data_size == 256 * 1024 * 1024);
  assert(test_cxlmem.pmem_size == 768 * 1024 * 1024);
}
