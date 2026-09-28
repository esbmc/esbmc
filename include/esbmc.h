/*
 * SPDX-FileCopyrightText: 2025 Lucas Cordeiro, Jeremy Morse, Bernd Fischer, Mikhail Ramalho
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * User-facing names for ESBMC's verification intrinsics.
 *
 * The intrinsics themselves are spelled __ESBMC_*, a reserved namespace
 * appropriate for names ESBMC's own translation generates. This header is the
 * supported surface for writing harnesses and stubs by hand, and the reference
 * for what may be called from verified code.
 *
 * The names are macros rather than declarations, and the mapping lives here
 * rather than in the compiler arguments ESBMC passes to clang. A global
 * -DESBMC_assert=__ESBMC_assert would rewrite that identifier in every
 * translation unit, including one that never includes this header and defines
 * an ESBMC_assert of its own -- ESBMC_* is not a reserved namespace, unlike the
 * __builtin_* rewrites next to it in build_compiler_args. Including this header
 * is opt-in, so the collision is visible where it happens.
 */

#ifndef __ESBMC_H
#define __ESBMC_H

#ifndef __ESBMC_execution
#error "esbmc.h can only be used for ESBMC runs"
#endif

/* Constrain the search space: paths reaching a false cond are not explored.
 * An assumption that is false on every path makes verification vacuous, so
 * pair a strong one with a reachability check. */
#define ESBMC_assume(cond) __ESBMC_assume((cond))

/* Assert cond, reporting msg when it is violated. */
#define ESBMC_assert(cond, msg) __ESBMC_assert((cond), (msg))

/* No ESBMC_cover: __ESBMC_cover is declared for C but implemented only in the
 * Python frontend, so a C call to it aborts with "Function call to
 * non-intrinsic prefixed with __ESBMC". */

/* Allocate size bytes with the lifetime of the enclosing function. */
#define ESBMC_alloca(size) __ESBMC_alloca((size))

/* True when both pointers address the same object, whatever their offsets. */
#define ESBMC_same_object(p, q) __ESBMC_same_object((p), (q))

/* Assert that control never reaches here. Needs
 * --enable-unreachability-intrinsic; without it the claim is suppressed. */
#define ESBMC_unreachable() __ESBMC_unreachable()

/* Bound the enclosing loop to n iterations. */
#define ESBMC_unroll(n) __ESBMC_unroll((n))

/* Run the enclosed region without a context switch. */
#define ESBMC_atomic_begin() __ESBMC_atomic_begin()
#define ESBMC_atomic_end() __ESBMC_atomic_end()

/* Offer the scheduler a context switch at this point. */
#define ESBMC_yield() __ESBMC_yield()

#endif /* __ESBMC_H */
