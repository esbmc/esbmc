/* Solidity dynamic array operations — inline-header design.
 *
 * Each dynamic array allocation reserves sizeof(size_t) bytes BEFORE
 * the data for an inline length header.  All length queries are O(1)
 * pointer arithmetic — no global lookup table, no loops, immune to
 * --unwind truncation.
 *
 * Memory layout:
 *   [ size_t length | element[0] | element[1] | ... ]
 *                    ^
 *                    returned pointer (what Solidity code sees)
 */
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include "solidity_types.h"

/* Legacy globals — kept as zero-initialized dummies so that any old
 * GOTO code referencing them still links, but they are unused. */
__attribute__((annotate("__ESBMC_inf_size"))) void *esbmc_array_ptrs[1];
__attribute__((annotate("__ESBMC_inf_size"))) size_t esbmc_array_lengths[1];
unsigned int esbmc_array_count;

/* T1.1 Stage S5: addr-replicated XOR fold for per-instance dyn-array
 * element keying.  See the function-body comment below for the design.
 * Same uint64 return shape as the prior S2 helper; no signature change.
 */
uint64_t _ESBMC_dynarr_idx(address_t addr, uint256_t idx)
{
__ESBMC_HIDE:;
  /* T1.1 Stage S5: addr-replicated XOR fold.
   *
   * Replaces the prior S2 pure XOR fold which was collision-vulnerable:
   * the solver could pick adversarial nondet addresses such that
   * `fold(clone.addr, 1) == fold(base.addr, 0)`, breaking
   * `esol_clone_dynamic_array_pass`.
   *
   * New encoding: `((addr-fold << 32) + addr-fold) ^ idx-fold`.  The
   * `* 0x100000001` (= `<<32 | id`) replicates the address fold into
   * the upper 32 bits of the slot key, while the index fold stays
   * concentrated in the lower bits.  For the realistic case
   * `addr < 2^32` (guaranteed by `(address_t)nondet_uint()` in
   * `_ESBMC_get_unique_address`) and `idx < 2^32` (real Solidity
   * dyn-array sizes are gas-bounded orders of magnitude below this):
   *   upper 32 bits of result = addr-fold (addr-determined)
   *   lower 32 bits of result = addr-fold ^ idx-fold
   * Two distinct (addr, idx) pairs cannot collide: equal upper bits
   * forces addr-fold equal, then equal lower bits forces idx-fold
   * equal — contradiction with distinctness.
   *
   * This survives Bitwuzla's array decision procedure where simpler
   * shift-or packs trigger "Equality over constant arrays not fully
   * supported yet".  The XOR-and-multiply structure keeps the per-
   * (addr, idx) Hamming-distance profile low for incremental pushes,
   * which Bitwuzla can simplify efficiently. */
  uint64_t a = (uint64_t)addr ^ (uint64_t)(addr >> 32) ^
               (uint64_t)(addr >> 64) ^ (uint64_t)(addr >> 96) ^
               (uint64_t)(addr >> 128);
  uint64_t i = (uint64_t)idx ^ (uint64_t)(idx >> 64) ^ (uint64_t)(idx >> 128) ^
               (uint64_t)(idx >> 192);
  return (a * 0x100000001ULL) ^ i;
}

/* ------------------------------------------------------------------ */
/*  Assertion helpers                                                  */
/* ------------------------------------------------------------------ */

void _ESBMC_array_null_check(int ok)
{
__ESBMC_HIDE:;
  if (!ok)
    assert(!"Null Array Pointer");
}

void _ESBMC_element_null_check(int ok)
{
__ESBMC_HIDE:;
  if (!ok)
    assert(!"Null Element Pointer");
}

void _ESBMC_zero_size_check(int ok)
{
__ESBMC_HIDE:;
  if (!ok)
    assert(!"Zero Element Size");
}

void _ESBMC_pop_empty_check(int ok)
{
__ESBMC_HIDE:;
  if (!ok)
    assert(!"Pop From Empty Array");
}

/* ------------------------------------------------------------------ */
/*  Inline-header helpers (private)                                   */
/* ------------------------------------------------------------------ */

/* Read the length stored just before the data pointer. */
static size_t _hdr_read(void *array)
{
  return ((size_t *)array)[-1];
}

/* Write the length stored just before the data pointer. */
static void _hdr_write(void *array, size_t length)
{
  ((size_t *)array)[-1] = length;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                        */
/* ------------------------------------------------------------------ */

/* Allocate a dynamic array with an inline length header (zero-init).
 * Returns a pointer to the DATA region (past the header).
 * The caller sees array[0..count-1]; the header is hidden.
 *
 * This variant uses calloc, so the data region is zero-initialized.
 * Use this when `count` is a COMPILE-TIME CONSTANT. For a symbolic
 * `count`, use _ESBMC_alloc_array_sym — calloc's internal memset loop
 * creates a VLA-typed backing (signed char [symbolic_size]) that loses
 * indexed write/read tracking through pointer reassignment. */
void *_ESBMC_alloc_array(size_t count, size_t elem_size)
{
__ESBMC_HIDE:;
  _ESBMC_zero_size_check(elem_size != 0);
  size_t total = sizeof(size_t) + count * elem_size;
  size_t *block = (size_t *)calloc(1, total);
  /* Library contract: this helper returns a valid pointer. Without the
   * assume, VSA records both the real alloc and the calloc-fails-returns-
   * NULL branch, so downstream `block + 1` evaluates to `<0, 8, void>` in
   * the points-to set, then propagates through every contract-struct
   * pointer field initialised via this helper. Callers like
   * `_ESBMC_arrcpy_2d` that start with `_ESBMC_element_null_check
   * (from_array != 0)` then see the null branch as SAT on reads like
   * `base->arr[0]` even though no real path produces it. Paired with
   * VSA's new ASSUME handler (see value_sett::apply_assume) this collapses
   * the spurious branch at the source. */
  __ESBMC_assume(block != 0);
  block[0] = count;           /* header = length */
  return (void *)(block + 1); /* data pointer    */
}

/* Symbolic-size variant: uses malloc + a TYPED-element zero loop to
 * initialize the data region.  We avoid `calloc(1, total)` because
 * calloc's internal memset lowering creates a VLA-typed temporary
 * (signed char [symbolic_size]) that loses indexed write/read tracking
 * through pointer reassignment.  An explicit typed-element loop has
 * no VLA temporary — the backing stays a flat pointer that symex
 * tracks cleanly through `<arr>[i]` reassigns.
 *
 * Real Solidity zero-initializes memory dyn-arrays, so this matches
 * source semantics.  Loop is `--unwind`-bounded like `_ESBMC_arrcpy`
 * — at `--unwind N`, only the first N elements are zeroed; callers
 * with longer arrays raise the bound explicitly. */
void *_ESBMC_alloc_array_sym(size_t count, size_t elem_size)
{
__ESBMC_HIDE:;
  _ESBMC_zero_size_check(elem_size != 0);
  size_t total = sizeof(size_t) + count * elem_size;
  size_t *block = (size_t *)malloc(total);
  /* Same non-null contract as `_ESBMC_alloc_array`; see comment there. */
  __ESBMC_assume(block != 0);
  block[0] = count;                 /* header = length */
  void *data = (void *)(block + 1); /* data pointer    */

  /* T1.1 Stage S5.5: zero-init data region via typed-element loop
   * (mirror of `_ESBMC_arrcpy`'s typed branches at lines 212-225). */
  if (elem_size == sizeof(uint256_t))
  {
    uint256_t *typed = (uint256_t *)data;
    for (size_t i = 0; i < count; i++)
      typed[i] = 0;
  }
  else if (elem_size == sizeof(int256_t))
  {
    int256_t *typed = (int256_t *)data;
    for (size_t i = 0; i < count; i++)
      typed[i] = 0;
  }
  else
  {
    char *bytes = (char *)data;
    for (size_t i = 0; i < count * elem_size; i++)
      bytes[i] = 0;
  }
  return data;
}

/* Build a 2D nested array backing for the verification harness:
 * outer pointer of `outer` slots, each slot a fresh inner pointer of
 * `inner` elements of `elem_size` bytes. Used by assign_param_nondet to
 * give Nondet_Extcall_<C>-dispatched entry points a valid (rather than
 * nil) pointer for `T[a][b]` / `T[a][]` / `T[][a]` / `T[][]` parameters,
 * so any in-body element read/write or arrcpy lands on a real buffer
 * with the declared length headers in place. Contents stay zero-init
 * — sound for "external caller passed some array of declared shape". */
void *_ESBMC_alloc_nested_2d(size_t outer, size_t inner, size_t elem_size)
{
__ESBMC_HIDE:;
  _ESBMC_zero_size_check(elem_size != 0);
  void **outer_buf = (void **)_ESBMC_alloc_array(outer, sizeof(void *));
  for (size_t i = 0; i < outer; i++)
    outer_buf[i] = _ESBMC_alloc_array(inner, elem_size);
  return (void *)outer_buf;
}

/* Update the length of an already-allocated array.
 * O(1) — just writes the header word. */
void _ESBMC_store_array(void *array, size_t length)
{
__ESBMC_HIDE:;
  _ESBMC_array_null_check(array != 0);
  _hdr_write(array, length);
}

/* Read the length of an allocated array.
 * O(1) — just reads the header word. */
unsigned int _ESBMC_array_length(void *array)
{
__ESBMC_HIDE:;
  if (array == NULL)
    return 0;
  return (unsigned int)_hdr_read(array);
}

/* Deep-copy an array into a fresh allocation (with its own header).
 * Used for memory→storage assignment and struct-return value copies. */
void *_ESBMC_arrcpy(void *from_array, size_t from_size, size_t size_of)
{
__ESBMC_HIDE:;
  /* Same VSA non-null hint as `_ESBMC_arrcpy_2d`: pairs with the ASSUME
   * handler in value_sett::apply_assume to strip spurious null branches
   * from the value-set that propagate from upstream NONDET struct inits. */
  __ESBMC_assume(from_array != 0);
  _ESBMC_element_null_check(from_array != 0);
  _ESBMC_zero_size_check(size_of != 0);
  _ESBMC_zero_size_check(from_size != 0);

  void *to_array = _ESBMC_alloc_array(from_size, size_of);

  /* Element-level copy for uint256/int256 (one SSA op per element).
   * Falls back to memcpy for other sizes. */
  if (size_of == sizeof(uint256_t))
  {
    uint256_t *src = (uint256_t *)from_array;
    uint256_t *dst = (uint256_t *)to_array;
    for (size_t i = 0; i < from_size; i++)
      dst[i] = src[i];
  }
  else if (size_of == sizeof(int256_t))
  {
    int256_t *src = (int256_t *)from_array;
    int256_t *dst = (int256_t *)to_array;
    for (size_t i = 0; i < from_size; i++)
      dst[i] = src[i];
  }
  else
  {
    __builtin_memcpy(to_array, from_array, from_size * size_of);
  }

  return to_array;
}

/* Deep-copy a 2D fixed-layout array: outer is `outer` pointers, each pointing
 * to an `inner`-element array of `elem_size`-byte elements.  Allocates a
 * fresh outer buffer and a fresh inner buffer per slot, then element-copies
 * base's inner rows into the new inner buffers.
 *
 * Needed because the alternative — frontend-emitted per-slot
 * `c->grid[i] = _ESBMC_arrcpy(base->grid[i], inner, elem_size)` after
 * `c->grid = _ESBMC_alloc_array(outer, sizeof(void*))` — confuses symex's
 * value-set tracking: successive index writes to a freshly-reassigned
 * pointer field don't flow through to subsequent reads.  Wrapping the
 * whole allocate+fill dance in a single C helper keeps all writes
 * inside one function frame where symex handles them cleanly.
 * (See `esol_clone_multi_dim_knownbug` for the original repro.) */
void *
_ESBMC_arrcpy_2d(void *from_array, size_t outer, size_t inner, size_t elem_size)
{
__ESBMC_HIDE:;
  /* VSA hint: value-set analysis otherwise admits null as a possible value
   * for `from_array` via the `*` (any-object) entry that propagates from
   * upstream NONDET struct initialisation during e.g. cpp_new + ctor +
   * struct-copy sequences. The assume prunes null at this entry point so
   * the subsequent null-check assert does not fire spuriously. Requires
   * the VSA ASSUME handler (value_sett::apply_assume). */
  __ESBMC_assume(from_array != 0);
  _ESBMC_element_null_check(from_array != 0);
  _ESBMC_zero_size_check(outer != 0);
  _ESBMC_zero_size_check(inner != 0);
  _ESBMC_zero_size_check(elem_size != 0);

  /* Step 1: byte-copy the outer pointer array via memcpy (the same
   * pattern _ESBMC_arrcpy uses for non-u256/int256 elements).  This
   * gives us a local `dst_outer` whose slots are the same pointer
   * values as `from_array`'s slots — and symex tracks the memcpy
   * byte-for-byte, so subsequent pointer reads through `dst_outer[i]`
   * see the original heap pointers cleanly. */
  void *dst_outer_raw = _ESBMC_alloc_array(outer, sizeof(void *));
  __builtin_memcpy(dst_outer_raw, from_array, outer * sizeof(void *));

  /* Step 2: replace each slot with a fresh arrcpy of its inner row. */
  void **dst_outer = (void **)dst_outer_raw;
  for (size_t i = 0; i < outer; i++)
    dst_outer[i] = _ESBMC_arrcpy(dst_outer[i], inner, elem_size);
  return (void *)dst_outer;
}

/* Append one element to the end of the array.
 * Returns a (possibly relocated) data pointer. */
void *_ESBMC_array_push(void *array, void *element, size_t size_of_element)
{
__ESBMC_HIDE:;
  _ESBMC_zero_size_check(size_of_element != 0);

  /* Phase 9 SSA-cleanup: callers (Solidity-frontend convert_ref.cpp,
   * the no-arg-push path's `_tmpzero` aux, the with-arg path's
   * stack-local backing) ALWAYS pass non-NULL element. Without an
   * explicit assume, symex explores the dead NULL-fallback branch,
   * which calloc'd a zero buffer (`dynamic_5_array`) and emitted
   * `element = (element == NULL) ? fallback_zero : element` as an
   * SSA ITE. This ITE turned every memcpy SAME-OBJECT chain at the
   * push call site into a two-candidate conflation (caller's stack-
   * local AND `dynamic_5_array`), and SMT picked whichever broke
   * the assertion. The frontend invariant is: element is never NULL
   * — assume it. */
  __ESBMC_assume(element != 0);

  if (array == NULL)
  {
    /* First push — allocate a fresh 1-element array. */
    void *new_array = _ESBMC_alloc_array(1, size_of_element);
    __builtin_memcpy(new_array, element, size_of_element);
    return new_array;
  }

  size_t old_len = _hdr_read(array);
  size_t new_len = old_len + 1;

  /* Grow: realloc the block (header + data). */
  size_t *old_block = ((size_t *)array) - 1;
  size_t *new_block =
    (size_t *)realloc(old_block, sizeof(size_t) + new_len * size_of_element);
  /* Soundness contract — match _ESBMC_alloc_array (line 137),
   * _ESBMC_alloc_array_sym (line 161), _ESBMC_arrcpy (line 233),
   * _ESBMC_array_push_uint256 (line 401). Without this, the SMT
   * model is free to pick the realloc-fails branch and propagate
   * a nondet-null return through `new_data = new_block + 1`,
   * which back-writes garbage into the caller's slot. */
  __ESBMC_assume(new_block != 0);
  new_block[0] = new_len;
  void *new_data = (void *)(new_block + 1);

  /* Copy the new element into the last slot. */
  __builtin_memcpy(
    (char *)new_data + old_len * size_of_element, element, size_of_element);

  return new_data;
}

/* Typed uint256-element push for mapping-of-dynarray write-through.
 *
 * Emitted by the Solidity frontend for `m[k].push(x)` where `m` has
 * `mapping(K => uint256[])`-shaped value. The generic `_ESBMC_array_push`
 * goes through `__builtin_memcpy` → `__ESBMC_memcpy` intrinsic, which
 * falls back to `__memcpy_impl`'s byte-loop (string.c:278) whenever `n`
 * is non-constant. Under `--unwind N --no-unwinding-assertions` that
 * loop is silently truncated, killing every post-push path. The typed
 * single-assignment below emits one SSA store per element and is not
 * subject to loop unwinding on the last-element write.
 *
 * Why a dedicated helper instead of rewriting `_ESBMC_array_push`:
 * switching the generic push to typed copies breaks six existing tests
 * (`dangling_ref_1`, `esol_clone_dynamic_array_*`, `nested_array_*`)
 * whose current "SUCCESSFUL" outcome is itself a side effect of loop
 * truncation killing post-push paths. Exposing the real post-push
 * semantics reveals assertion failures that need separate investigation.
 * Until those are fixed, the two helpers coexist. */
void *_ESBMC_array_push_uint256(void *array, uint256_t element)
{
__ESBMC_HIDE:;
  size_t old_len = (array != NULL) ? _hdr_read(array) : 0;
  size_t new_len = old_len + 1;

  /* Fresh malloc + typed single-assign — no memcpy, no copy loop, no
   * realloc. Critical trade-off: old elements (slots 0..old_len-1)
   * are NOT carried over from the previous allocation; only the
   * header and slot [old_len] are definite. Accesses to those
   * stale-slot indices after a push yield nondet, which is a sound
   * over-approximation (the solver simply can't refute anything that
   * would require reading a pre-push element).
   *
   * Rationale: avoiding the copy is the only way to keep this helper
   * free of unwinding-bound-sensitive loops. Any typed copy across
   * old_len with a symbolic `old_len` introduces a loop that
   * `--unwind N --no-unwinding-assertions` silently truncates,
   * reinstating the path-death hazard the helper is supposed to fix.
   * Realloc's internal copy-128 loop is even worse — each iteration
   * materialises an individual SSA copy, blowing up the formula.
   *
   * For the mapping-of-dynarray write-through shipped alongside this
   * helper (regressions `map_dynarr_*`), the push is the only mutator
   * and element reads target the freshest slot, so the stale-slot
   * approximation is harmless. Extending the helper to preserve old
   * elements requires moving the model away from heap pointers — see
   * `#sol_dynarray_state` for the state-var-infinite-array pattern. */
  size_t *new_block =
    (size_t *)malloc(sizeof(size_t) + new_len * sizeof(uint256_t));
  __ESBMC_assume(new_block != 0);
  new_block[0] = new_len;
  uint256_t *new_data = (uint256_t *)(new_block + 1);

  /* Typed single-assignment for the new last element. */
  new_data[old_len] = element;

  return (void *)new_data;
}

/* Remove the last element from the array. */
void _ESBMC_array_pop(void *array, size_t size_of_element)
{
__ESBMC_HIDE:;
  _ESBMC_array_null_check(array != 0);
  _ESBMC_zero_size_check(size_of_element != 0);

  size_t len = _hdr_read(array);
  _ESBMC_pop_empty_check(len > 0);

  size_t new_len = len - 1;
  _hdr_write(array, new_len);

  /* Optionally shrink (realloc). Skipped for simplicity —
   * the extra capacity is harmless in verification. */
}
