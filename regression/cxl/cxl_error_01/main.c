// CXL error injection and handling test.
// Tests that the driver correctly handles CXL error types:
// correctable, non-fatal, and fatal errors.
// Expected: VERIFICATION SUCCESSFUL

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

/* CXL error types */
enum cxl_error_type {
  CXL_ERROR_CORRECTABLE = 0,
  CXL_ERROR_NON_FATAL,
  CXL_ERROR_FATAL,
};

/* CXL device error state */
struct cxl_dev {
  enum cxl_error_type last_error;
  uint64_t correctable_count;
  uint64_t non_fatal_count;
  uint64_t fatal_count;
  int system_dead;
};

static struct cxl_dev test_cxld;

/* Simulated error injection */
void cxl_inject_error(struct cxl_dev *cxld, enum cxl_error_type type)
{
  assert(cxld != NULL);

  cxld->last_error = type;

  switch (type)
  {
  case CXL_ERROR_CORRECTABLE:
    cxld->correctable_count++;
    /* Correctable: driver logs and continues */
    break;

  case CXL_ERROR_NON_FATAL:
    cxld->non_fatal_count++;
    /* Non-fatal: driver attempts recovery */
    break;

  case CXL_ERROR_FATAL:
    cxld->fatal_count++;
    /* Fatal: driver marks system as dead */
    cxld->system_dead = 1;
    break;
  }
}

/* Error handler: process error and take appropriate action */
int cxl_error_handler(struct cxl_dev *cxld)
{
  assert(cxld != NULL);

  switch (cxld->last_error)
  {
  case CXL_ERROR_CORRECTABLE:
    /* Log correctable error, continue operation */
    return 0;

  case CXL_ERROR_NON_FATAL:
    /* Attempt recovery: reset device */
    cxld->last_error = CXL_ERROR_CORRECTABLE;
    return 0;

  case CXL_ERROR_FATAL:
    /* Fatal error: cannot recover */
    return -1;

  default:
    return 0;
  }
}

int main()
{
  test_cxld.system_dead = 0;
  test_cxld.correctable_count = 0;
  test_cxld.non_fatal_count = 0;
  test_cxld.fatal_count = 0;

  /* Test 1: Correctable error */
  cxl_inject_error(&test_cxld, CXL_ERROR_CORRECTABLE);
  assert(cxl_error_handler(&test_cxld) == 0);
  assert(test_cxld.correctable_count == 1);
  assert(test_cxld.system_dead == 0);

  /* Test 2: Non-fatal error */
  cxl_inject_error(&test_cxld, CXL_ERROR_NON_FATAL);
  assert(cxl_error_handler(&test_cxld) == 0);
  assert(test_cxld.non_fatal_count == 1);
  assert(test_cxld.system_dead == 0);

  /* Test 3: Fatal error */
  cxl_inject_error(&test_cxld, CXL_ERROR_FATAL);
  assert(cxl_error_handler(&test_cxld) == -1);
  assert(test_cxld.fatal_count == 1);
  assert(test_cxld.system_dead == 1);

  /* Verify error counts */
  assert(test_cxld.correctable_count == 1);
  assert(test_cxld.non_fatal_count == 1);
  assert(test_cxld.fatal_count == 1);
}
