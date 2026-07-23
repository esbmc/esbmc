// CXL mailbox command queue state machine test.
// Tests that the mailbox command queue state transitions are correct:
// IDLE -> BUSY -> COMPLETE -> IDLE.
// Expected: VERIFICATION SUCCESSFUL

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

/* Mailbox command queue state */
enum mbox_state {
  MBOX_IDLE = 0,
  MBOX_BUSY,
  MBOX_COMPLETE,
};

struct cxl_dev {
  enum mbox_state state;
  uint16_t pending_opcode;
  uint32_t response_status;
  int command_count;
};

static struct cxl_dev test_cxld;

/* Submit a mailbox command (non-blocking) */
int cxl_mailbox_submit(struct cxl_dev *cxld, uint16_t opcode)
{
  assert(cxld != NULL);
  /* Must be idle to submit */
  assert(cxld->state == MBOX_IDLE);

  cxld->pending_opcode = opcode;
  cxld->state = MBOX_BUSY;
  cxld->command_count++;
  return 0;
}

/* Check if command is complete */
int cxl_mailbox_is_complete(struct cxl_dev *cxld)
{
  assert(cxld != NULL);
  return cxld->state == MBOX_COMPLETE;
}

/* Get response status (only valid when complete) */
uint32_t cxl_mailbox_get_response(struct cxl_dev *cxld)
{
  assert(cxld != NULL);
  assert(cxld->state == MBOX_COMPLETE);
  return cxld->response_status;
}

/* Hardware completes the pending command */
void cxl_mailbox_hw_complete(struct cxl_dev *cxld, uint32_t status)
{
  assert(cxld != NULL);
  assert(cxld->state == MBOX_BUSY);

  cxld->response_status = status;
  cxld->state = MBOX_COMPLETE;
}

/* Driver acknowledges completion and returns to idle */
void cxl_mailbox_ack_complete(struct cxl_dev *cxld)
{
  assert(cxld != NULL);
  assert(cxld->state == MBOX_COMPLETE);

  cxld->state = MBOX_IDLE;
  cxld->pending_opcode = 0;
  cxld->response_status = 0;
}

int main()
{
  test_cxld.state = MBOX_IDLE;
  test_cxld.command_count = 0;

  /* Step 1: Submit command */
  int ret = cxl_mailbox_submit(&test_cxld, 0x0001);
  assert(ret == 0);
  assert(test_cxld.state == MBOX_BUSY);
  assert(test_cxld.command_count == 1);

  /* Step 2: Check not complete yet */
  assert(cxl_mailbox_is_complete(&test_cxld) == 0);

  /* Step 3: Hardware completes the command */
  cxl_mailbox_hw_complete(&test_cxld, 0);

  /* Step 4: Check complete */
  assert(cxl_mailbox_is_complete(&test_cxld) == 1);
  assert(cxl_mailbox_get_response(&test_cxld) == 0);

  /* Step 5: Driver acknowledges */
  cxl_mailbox_ack_complete(&test_cxld);
  assert(test_cxld.state == MBOX_IDLE);

  /* Step 6: Submit second command */
  ret = cxl_mailbox_submit(&test_cxld, 0x0002);
  assert(ret == 0);
  assert(test_cxld.state == MBOX_BUSY);
  assert(test_cxld.command_count == 2);

  /* Step 7: Hardware completes with error */
  cxl_mailbox_hw_complete(&test_cxld, 1); /* error status */
  assert(cxl_mailbox_get_response(&test_cxld) == 1);

  /* Step 8: Acknowledge */
  cxl_mailbox_ack_complete(&test_cxld);
  assert(test_cxld.state == MBOX_IDLE);

  /* Verify state machine invariant: never skip states */
  /* IDLE -> BUSY -> COMPLETE -> IDLE */
  assert(test_cxld.command_count == 2);
}
