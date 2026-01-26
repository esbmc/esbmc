/* =============================================================================
 * TEST: Union Inside Struct — Pillar 1 (Access Path Restoration)
 * =============================================================================
 *
 * PURPOSE:
 *   Test field-sensitive assigns when a struct contains a union member.
 *   Union members share the same base offset, so the path restorer must
 *   correctly handle the union's zero-offset semantics while still
 *   distinguishing the outer struct fields.
 *
 * REAL-WORLD RELEVANCE:
 *   Hardware register files in embedded systems often use unions for
 *   bitfield access vs. raw word access. Verifying driver code requires
 *   tracking which view of the union was modified.
 *
 * EXPECTED: VERIFICATION SUCCESSFUL
 * =========================================================================== */

#include <assert.h>
#include <stddef.h>

typedef union {
    int as_int;
    float as_float;
} DataValue;

typedef struct {
    int tag;          /* Type discriminator */
    DataValue data;   /* Union payload */
    int sequence_num; /* Metadata */
} TaggedData;

/* Modifies only the tag and the union payload, preserving sequence_num */
void set_int_value(TaggedData *td, int val) {
    __ESBMC_requires(td != NULL);
    __ESBMC_assigns(td->tag, td->data);
    __ESBMC_ensures(td->tag == 1);
    __ESBMC_ensures(td->data.as_int == val);
    __ESBMC_ensures(td->sequence_num == __ESBMC_old(td->sequence_num));

    td->tag = 1;
    td->data.as_int = val;
}

int main() {
    TaggedData td;
    td.tag = 0;
    td.data.as_int = 0;
    td.sequence_num = 42;

    set_int_value(&td, 999);

    assert(td.tag == 1);
    assert(td.data.as_int == 999);
    assert(td.sequence_num == 42);  /* Not in assigns — preserved */

    return 0;
}
