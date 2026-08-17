// CXL mailbox command verification test.
// Tests that mailbox commands are submitted correctly and that the driver
// checks the command status before using the output payload.
// Expected: VERIFICATION FAILED (driver bug: uses output without checking status)

#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

/* CXL register offsets */
#define CXL_REGMAP_MAILBOX   0x0100
#define CXL_MBOX_OP_GET_CAPABILITIES    0x0002

/* CXL device structure (minimal) */
struct cxl_dev {
  void *regs;
};

/* Mailbox command */
struct cxl_mailbox_cmd {
  uint16_t opcode;
  uint16_t payload_in_size;
  uint16_t payload_out_size;
  void *payload_in;
  void *payload_out;
  uint32_t status;
};

/* Simulated CXL device */
static struct cxl_dev test_cxld;
static uint32_t test_mailbox_status;

/* Override the mailbox model for deterministic testing */
int cxl_mailbox_send_cmd(struct cxl_dev *cxld, struct cxl_mailbox_cmd *cmd)
{
  (void)cxld;
  /* Simulate a failed command */
  cmd->status = 1; /* non-zero = error */
  return -EIO;
}

/*
 * BUG: This driver code does NOT check the return value of
 * cxl_mailbox_send_cmd() before using the output payload.
 * This is a real-world bug pattern in CXL drivers.
 */
int get_cxl_capabilities(struct cxl_dev *cxld)
{
  struct cxl_mailbox_cmd cmd;
  uint8_t output_buf[256];

  memset(&cmd, 0, sizeof(cmd));
  cmd.opcode = CXL_MBOX_OP_GET_CAPABILITIES;
  cmd.payload_out = output_buf;
  cmd.payload_out_size = sizeof(output_buf);

  /* BUG: Missing return value check! */
  cxl_mailbox_send_cmd(cxld, &cmd);

  /*
   * Using output without checking status — if the command failed,
   * output_buf contains garbage.  This assertion should fail
   * when the model returns a non-zero status.
   */
  assert(cmd.status == 0);

  return 0;
}

int main()
{
  test_cxld.regs = (void *)0x1000;

  int ret = get_cxl_capabilities(&test_cxld);
  (void)ret;
}
