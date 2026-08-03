// CDAT table validation: checksum over a bounded buffer, and per-entry length
// checks before the entry is read.
//
// The synthetic counterpart to regression/cxl-linux/harness_cdat_checksum,
// which proves the same bound against the real cdat_checksum() in
// drivers/cxl/core/pci.c. Both say: the checksum length is the caller's
// obligation, not the table's.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

extern unsigned char __VERIFIER_nondet_uchar(void);
extern unsigned short __VERIFIER_nondet_ushort(void);

#define CDAT_BUF_LEN 64
#define DSMAS_SIZE (sizeof(struct acpi_cdat_header) + sizeof(struct acpi_cdat_dsmas))

int main()
{
  unsigned char buf[CDAT_BUF_LEN];
  size_t len = (size_t)__VERIFIER_nondet_uchar();

  /* read_cdat_data() allocates for the length it asked for; the length it
     then checksums must not exceed that. This is the bound the real harness
     proves and its _fail variant relaxes. */
  __ESBMC_assume(len <= sizeof(buf));
  memset(buf, 0, sizeof(buf));
  assert(cdat_checksum(buf, len) == 0);

  /* A table sums to zero exactly when it is well-formed; flipping one byte
     is enough to break that, and the checksum must notice. */
  if (len > 0)
  {
    buf[0] = 0x01;
    assert(cdat_checksum(buf, len) == 0x01);
    buf[0] = 0x00;
  }

  /* Entry validation: an entry is accepted only at exactly its type's size,
     and only if it lies wholly inside the table. */
  struct acpi_cdat_header *hdr = (struct acpi_cdat_header *)buf;
  const void *end = buf + sizeof(buf);

  hdr->type = ACPI_CDAT_TYPE_DSMAS;
  hdr->reserved = 0;
  hdr->length = (u16)DSMAS_SIZE;
  assert(cdat_entry_validate(hdr, DSMAS_SIZE, end) == 0);

  /* Any other length is malformed, whatever the table claims. */
  u16 claimed = __VERIFIER_nondet_ushort();
  hdr->length = claimed;
  if (cdat_entry_validate(hdr, DSMAS_SIZE, end) == 0)
  {
    assert(claimed == DSMAS_SIZE);
    assert((char *)hdr + claimed <= (char *)end);
  }

  /* An entry near the end of the table is rejected even at the right size:
     the length check alone would let it run off. */
  struct acpi_cdat_header *tail =
    (struct acpi_cdat_header *)(buf + sizeof(buf) - 4);
  tail->length = (u16)DSMAS_SIZE;
  assert(cdat_entry_validate(tail, DSMAS_SIZE, end) == -EINVAL);

  return 0;
}
