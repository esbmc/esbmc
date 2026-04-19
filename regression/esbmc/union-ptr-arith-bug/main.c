/**
 * ESBMC Bug: constant_union member extraction in simplify_expr2.cpp
 *
 * ROOT CAUSE (fixed):
 * In member2t::do_simplify(), when extracting a member from constant_union2t,
 * the code used the type's member index to look up the value in datatype_members.
 * But constant_union stores the value at position 0 with init_field indicating
 * which member was initialized.
 *
 * Example: union { s2 s; s2 r; } initialized with {0}
 * - Sets init_field = "s" (first member)
 * - datatype_members[0] = the value
 * - Reading .r looked for member index 1, but datatype_members only has index 0
 * - Simplification failed, leaving unsimplified member2t expressions
 * - These propagated through pointer arithmetic, causing wrong offsets
 *
 * SYMPTOM:
 * When computing &data[2] where data = (s4*)&h.memory[8], ESBMC produced
 * &h.memory[7] + 2 instead of &h.memory[16].
 *
 * FIX (in simplify_expr2.cpp member2t::do_simplify):
 * For constant_union, always read from datatype_members[0] since that's where
 * the value is stored, with a type compatibility check to avoid incorrect
 * simplification when reading a different member than was initialized.
 *
 * Run: esbmc esbmc_ptr_arith_bug.c --unwind 5
 * Expected: VERIFICATION SUCCESSFUL (after fix)
 */

#include <assert.h>
#include <stdbool.h>

typedef unsigned char u1;
typedef unsigned short u2;
typedef short s2;
typedef int s4;

/* Union type - used in stack slots */
typedef union slot { s2 s; s2 r; } slot_t;

/* Frame with array of unions - separate from heap */
typedef struct frame { slot_t stack[8]; } frame_t;

/*
 * KEY TRIGGER: struct with array-of-unions followed by another array.
 * Removing stack_types makes the bug disappear.
 */
typedef struct saved_frame {
    const u1* code;
    u2 code_length;
    u2 pc;
    slot_t stack[8];      /* Array of unions */
    u1 stack_types[8];    /* Regular array AFTER union array - triggers bug! */
} saved_frame_t;

/*
 * Heap contains array of saved_frame.
 * Having call_stack[] embedded is necessary to trigger the bug.
 */
typedef struct heap {
    u1 memory[256];
    u2 free_ptr;
    u2 object_table[16];
    u2 num_objects;
    u1 transaction_buffer[32];
    u2 transaction_ptr;
    bool in_transaction;
    void* classpool;
    saved_frame_t call_stack[1];  /* Embedded struct with union arrays */
    u1 call_depth;
} heap_t;

int main(void) {
    heap_t h = {0};
    frame_t f = {0};

    /* Setup: object 1 is at memory offset 0 */
    h.object_table[1] = 0;

    /* Store reference value in union */
    f.stack[0].r = 1;

    /* Read from union - ESBMC represents this as { .r=1 }.r */
    s2 ref = f.stack[0].r;

    /* Use union-derived value as index into object_table */
    u2 offset = h.object_table[ref];

    /* Compute s4* pointer: should be &h.memory[0 + 8] = &h.memory[8] */
    s4* data = (s4*)&h.memory[offset + 8];

    /* Index by 2: should give &h.memory[8 + 2*4] = &h.memory[16] */
    s4* elem = &data[2];

    /* BUG: ESBMC computes elem = &h.memory[7] + 2 instead of &h.memory[16] */
    *elem = 12345;

    return 0;
}
