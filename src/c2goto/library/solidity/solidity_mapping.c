/* Solidity mapping data structure */
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "solidity_types.h"

/* SMT-array-backed mapping storage.
 *
 * The previous linked-list `_ESBMC_Mapping` walked O(N) per get/set, which
 * made k-induction's per-iteration symex cost grow quadratically with the
 * number of distinct keys.  All slots now live in this single global; the
 * `(struct mapping_t).base` reinterpreted as uint64 + `addr` + `key` are
 * folded to a 64-bit slot index by `_ESBMC_map_idx` (full variant) or
 * `_ESBMC_map_idx_fast` (variant without `addr`).
 *
 * Default-zero semantics: ESBMC injects an `ASSIGN _ESBMC_map_storage={ 0 };`
 * at __ESBMC_main start (verified pattern for every __ESBMC_inf_size global
 * — see solidity_address.c's sol_addr_array etc. with `--goto-functions-only`),
 * which lowers to SMT `array_of(NULL)`.  `select` of a never-written slot
 * returns NULL and the existing `p ? *p : 0` guard in each per-type getter
 * delivers Solidity's default-zero semantics for free. */
__attribute__((annotate("__ESBMC_inf_size:480"))) void *_ESBMC_map_storage[1];

/* 480-bit composite key type: concat(mid_64, addr_160, key_256). */
typedef unsigned BIGINT(480) _ESBMC_map_key_t;

/* Stage-2-only placeholder: the frontend at solidity_convert_decl.cpp:935-944
 * still emits a per-state-var `__ESBMC_inf_size struct _ESBMC_Mapping
 * _ESBMC_inf_<name>[]` symbol per mapping.  We don't use those slots at runtime
 * — they exist solely so that the symbol table has a unique linker-assigned
 * address per state var, which we reinterpret as the `mid` ID in `map_get_raw`.
 * Stage 3 drops both the placeholder and the frontend allocation. */
struct _ESBMC_Mapping
{
  uint8_t _stage2_placeholder;
};

/* === SMT-array-backed mapping representation (Stage 3: explicit mid) ====
 *
 * The previous linked-list `_ESBMC_Mapping` walked O(N) per get/set, which
 * made k-induction's per-iteration symex cost grow quadratically with the
 * number of distinct keys.  All slots now live in the single global
 * `_ESBMC_map_storage` declared in the _fast section above; the index
 * compounds (mid, addr, key) into a 64-bit slot via `_ESBMC_map_idx`.
 *
 * `mapping_t` carries an explicit `uint64_t mid` (per-state-var unique
 * ID, frontend-assigned via `solidity_convertert::next_mapping_mid`) so
 * the runtime can read it directly with no SMT-side conversion overhead.
 * Stage 2 had cast `(uintptr_t)m->base` to derive the same ID; that cast
 * generated a pointer-to-uint64 SMT op at every access, non-trivial for
 * the solver to discharge.  The `base` field stays as a legacy unused
 * field so the frontend's existing `_ESBMC_inf_<name>` allocation path
 * (preserved as dead allocations during Stage 3) still type-checks.
 *
 * Per-instance keyspace isolation (clone semantics) is preserved: the
 * deep-copy walker in solidity_convert_constructor.cpp:1125 retargets
 * `m->addr` on the clone to a fresh nondet address.  Distinct `addr`
 * values hash to disjoint slot indices in `_ESBMC_map_storage`, so
 * post-clone writes on one instance don't interfere with the other.
 */
struct mapping_t
{
  void *base;           /* legacy unused (kept so existing frontend
                           `mapping_t = { &_ESBMC_inf_*, this->$address }`
                           init keeps type-checking; runtime ignores) */
  uint64_t mid;         /* per-state-var unique ID — frontend assigns
                           from solidity_convertert::next_mapping_mid */
  address_t addr : 160; /* clone keyspace partition */
} __attribute__((packed));

/* Fold (mid, addr, key) into a 64-bit slot index.  Pure XOR fold of
 * 64-bit lanes — keeps the SMT encoding lightweight (no multiplication,
 * no shifts of variables; only constant shifts which the solver evaluates
 * statically in bit-vector arithmetic).
 *
 * **Soundness:** this fold is NOT injective.  The earlier comment said
 * "Collision rate 2^-64 per pair is acceptable per existing precedent"
 * — formally void in BMC/SMT (the solver finds distinct triples with
 * equal fold).  Causes false aliasing in mapping storage.  Audit S0
 * (2026-04-30) regression-locked under
 * `mapping_idx_fold_collision_pass_knownbug` and ledger entry #22.
 * Closure requires 256-bit array-domain support across solvers. */
static inline _ESBMC_map_key_t
_ESBMC_map_idx(uint64_t mid, address_t addr, uint256_t key)
{
__ESBMC_HIDE:;
  _ESBMC_map_key_t result = (_ESBMC_map_key_t)mid;
  _ESBMC_map_key_t mask160 =
    (((_ESBMC_map_key_t)1) << 160) - (_ESBMC_map_key_t)1;
  result = (result << 160) | ((_ESBMC_map_key_t)addr & mask160);
  result = (result << 256) | (_ESBMC_map_key_t)key;
  return result;
}

void *map_get_raw(struct mapping_t *m, uint256_t key)
{
__ESBMC_HIDE:;
  return _ESBMC_map_storage[_ESBMC_map_idx(m->mid, m->addr, key)];
}

void map_set_raw(struct mapping_t *m, uint256_t key, void *val)
{
__ESBMC_HIDE:;
  _ESBMC_map_storage[_ESBMC_map_idx(m->mid, m->addr, key)] = val;
}

/* uint256_t */
void map_uint_set(struct mapping_t *m, uint256_t k, uint256_t v)
{
__ESBMC_HIDE:;
  uint256_t *p = (uint256_t *)malloc(sizeof *p);
  *p = v;
  map_set_raw(m, k, p);
}
uint256_t map_uint_get(struct mapping_t *m, uint256_t k)
{
__ESBMC_HIDE:;
  uint256_t *p = (uint256_t *)map_get_raw(m, k);
  return p ? *p : (uint256_t)0;
}

/* int256_t */
void map_int_set(struct mapping_t *m, uint256_t k, int256_t v)
{
__ESBMC_HIDE:;
  int256_t *p = (int256_t *)malloc(sizeof *p);
  *p = v;
  map_set_raw(m, k, p);
}
int256_t map_int_get(struct mapping_t *m, uint256_t k)
{
__ESBMC_HIDE:;
  int256_t *p = (int256_t *)map_get_raw(m, k);
  return p ? *p : (int256_t)0;
}

/* string */
void map_string_set(struct mapping_t *m, uint256_t k, char *v)
{
__ESBMC_HIDE:;
  char **p = (char **)malloc(sizeof *p);
  *p = v;
  map_set_raw(m, k, p);
}
char *map_string_get(struct mapping_t *m, uint256_t k)
{
__ESBMC_HIDE:;
  char **p = (char **)map_get_raw(m, k);
  return p ? *p : (char *)0;
}

/* bool */
void map_bool_set(struct mapping_t *m, uint256_t k, bool v)
{
__ESBMC_HIDE:;
  bool *p = (bool *)malloc(sizeof *p);
  *p = v;
  map_set_raw(m, k, p);
}

bool map_bool_get(struct mapping_t *m, uint256_t k)
{
__ESBMC_HIDE:;
  bool *p = (bool *)map_get_raw(m, k);
  return p ? *p : false;
}

/* generic */
void map_generic_set(struct mapping_t *m, uint256_t k, const void *v, size_t sz)
{
__ESBMC_HIDE:;
  void *p = malloc(sz);
  memcpy(p, v, sz);
  map_set_raw(m, k, p);
}
void *map_generic_get(struct mapping_t *m, uint256_t k)
{
__ESBMC_HIDE:;
  return map_get_raw(m, k);
}

/* dynarray — for mapping(K => T[]) value slots.
 *
 * Stores a pointer-to-pointer so the frontend can write back a relocated
 * data pointer after `_ESBMC_array_push_uint256` allocates a new slab.
 * Unlike `map_generic_*` (which copies the VALUE inline, wrong for
 * dynarray because the value IS a heap pointer that may change), this
 * pair preserves the pointer-identity across push/pop cycles.
 *
 * `map_dynarr_get` returns the currently-stored data pointer (or NULL
 * if the key has never been written). `map_dynarr_set` installs a new
 * data pointer. Both operate on a single `void *` payload, regardless
 * of the array's element type — the frontend is responsible for
 * element-typed load/store through the returned pointer. */
void map_dynarr_set(struct mapping_t *m, uint256_t k, void *arr)
{
__ESBMC_HIDE:;
  void **p = (void **)malloc(sizeof(void *));
  *p = arr;
  map_set_raw(m, k, p);
}
void *map_dynarr_get(struct mapping_t *m, uint256_t k)
{
__ESBMC_HIDE:;
  void **p = (void **)map_get_raw(m, k);
  return p ? *p : (void *)0;
}

/* fixed-size array — for mapping(K => T[N]) value slots.
 *
 * Solidity semantics: every key is conceptually pre-bound to a fresh
 * N-element zero-initialised array, so a first read of a never-written
 * key must yield a valid pointer (not NULL) and element writes via that
 * pointer must persist for subsequent reads with the same key.
 *
 * Differs from map_dynarr_* in two ways:
 *  - Allocation is one-shot (fixed size), not reallocating on push.
 *  - First get lazily allocates+zero-inits so the returned pointer is
 *    immediately indexable.
 *
 * sz is the total byte size of the T[N] slot (N * sizeof(T)). The
 * frontend computes sz from the array's element size and compile-time
 * extent; we treat the payload as an opaque byte slab here. */
void *map_fixed_arr_get(struct mapping_t *m, uint256_t k, size_t sz)
{
__ESBMC_HIDE:;
  void *p = map_get_raw(m, k);
  if (p)
    return p;
  void *data = calloc(1, sz);
  map_set_raw(m, k, data);
  return data;
}

/* === SMT-array-backed mapping representation (Stage 1: _fast variant) ====
 *
 * Per-state-var isolation: `mid` distinguishes different mapping state vars
 * that share the same `_ESBMC_map_storage` global (declared at the top of
 * this file).  The `_fast` variant has no `addr` field (single-instance
 * only); the full `_ESBMC_Mapping` variant adds `addr` for clone-keyspace
 * partitioning.
 */
struct mapping_t_fast
{
  uint64_t mid; /* per-state-var unique ID */
};

/* Fold (mid, key) into a 64-bit slot index.  See `_ESBMC_map_idx` above
 * for the rationale of the pure-XOR fold over multiplicative mixing. */
static inline _ESBMC_map_key_t _ESBMC_map_idx_fast(uint64_t mid, uint256_t key)
{
__ESBMC_HIDE:;
  _ESBMC_map_key_t result = (_ESBMC_map_key_t)mid;
  result = (result << 160) | (_ESBMC_map_key_t)0;
  result = (result << 256) | (_ESBMC_map_key_t)key;
  return result;
}

void *map_get_raw_fast(struct mapping_t_fast *m, uint256_t key)
{
__ESBMC_HIDE:;
  return _ESBMC_map_storage[_ESBMC_map_idx_fast(m->mid, key)];
}

void map_set_raw_fast(struct mapping_t_fast *m, uint256_t key, void *val)
{
__ESBMC_HIDE:;
  _ESBMC_map_storage[_ESBMC_map_idx_fast(m->mid, key)] = val;
}

/* uint256_t */
void map_uint_set_fast(struct mapping_t_fast *m, uint256_t k, uint256_t v)
{
__ESBMC_HIDE:;
  uint256_t *p = (uint256_t *)malloc(sizeof *p);
  *p = v;
  map_set_raw_fast(m, k, p);
}
uint256_t map_uint_get_fast(struct mapping_t_fast *m, uint256_t k)
{
__ESBMC_HIDE:;
  uint256_t *p = (uint256_t *)map_get_raw_fast(m, k);
  return p ? *p : (uint256_t)0;
}

/* int256_t */
void map_int_set_fast(struct mapping_t_fast *m, uint256_t k, int256_t v)
{
__ESBMC_HIDE:;
  int256_t *p = (int256_t *)malloc(sizeof *p);
  *p = v;
  map_set_raw_fast(m, k, p);
}
int256_t map_int_get_fast(struct mapping_t_fast *m, uint256_t k)
{
__ESBMC_HIDE:;
  int256_t *p = (int256_t *)map_get_raw_fast(m, k);
  return p ? *p : (int256_t)0;
}

/* string */
void map_string_set_fast(struct mapping_t_fast *m, uint256_t k, char *v)
{
__ESBMC_HIDE:;
  char **p = (char **)malloc(sizeof *p);
  *p = v;
  map_set_raw_fast(m, k, p);
}
char *map_string_get_fast(struct mapping_t_fast *m, uint256_t k)
{
__ESBMC_HIDE:;
  char **p = (char **)map_get_raw_fast(m, k);
  return p ? *p : (char *)0;
}

/* bool */
void map_bool_set_fast(struct mapping_t_fast *m, uint256_t k, bool v)
{
__ESBMC_HIDE:;
  bool *p = (bool *)malloc(sizeof *p);
  *p = v;
  map_set_raw_fast(m, k, p);
}
bool map_bool_get_fast(struct mapping_t_fast *m, uint256_t k)
{
__ESBMC_HIDE:;
  bool *p = (bool *)map_get_raw_fast(m, k);
  return p ? *p : false;
}

/* generic */
void map_generic_set_fast(
  struct mapping_t_fast *m,
  uint256_t k,
  const void *v,
  size_t sz)
{
__ESBMC_HIDE:;
  void *p = malloc(sz);
  memcpy(p, v, sz);
  map_set_raw_fast(m, k, p);
}
void *map_generic_get_fast(struct mapping_t_fast *m, uint256_t k)
{
__ESBMC_HIDE:;
  return map_get_raw_fast(m, k);
}
