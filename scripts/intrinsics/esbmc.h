#ifndef __ESBMC_H
#define __ESBMC_H

// This header must be executed by ESBMC
#ifndef __ESBMC_execution
#error "esbmc.h should only be used for ESBMC runs"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Adds a condition into the current execution path
 *
 * ESBMC_assume will reduce the state-space search for a given
 * program by adding `cond` as a path condition to any properties.
 *
 * int x; __ESBMC_assume(x > 10);
 * if(x <= 10)
 *      ... // unreachable
 *
 * @param cond boolean condition to be added into the current path condition
 */
void ESBMC_assume(_Bool cond);

/**
 * @brief Verifies whether a given property is correct
 *
 * ESBMC_assert will add `cond` as property that must always
 * hold for a program. If a path where `cond` is not met exists
 * then, ESBMC will generate and provide a Counter Example containing
 * all the states that lead to the property violation.
 *
 * Note 1: The C function assert is converted into a call of this function
 * Note 2: Asserts can hold vacuously
 *
 * @param cond boolean condition that must be met
 * @param msg report message to be shown at esbmc counter example
 */
void ESBMC_assert(_Bool cond, const char *msg);

/**
 * @brief Allocates a symbol into the stack
 *
 * ESBMC_alloca is used to generate a memory allocation into the stack
 * with symbolic length.
 *
 * Note: This operation always succeeds
 *
 * @param N size of the allocation
 */
void *ESBMC_alloca(unsigned int N);

/**
 * @brief Begins an atomic block
 *
 * Note: Must be paired with ESBMC_atomic_end
 *
 * An atomic block will stop any interleavings between
 * the block instructions.
 */
void ESBMC_atomic_begin();

/**
 * @brief End an atomic block
 *
 * Note: Must be paired with ESBMC_atomic_begin
 *
 * An atomic block will stop any interleavings between
 * the block instructions.
 */
void ESBMC_atomic_end();

/**
 * @brief Initializes a symbol or object nondeterministically
 *
 * ESBMC_init_object is used to initialize all members of an object
 * (e.g. memory location) or symbol (e.g. variable) with nondeterminstic
 * values.
 *
 * int x = 42;
 * ESBMC_init(x);
 * assert(x != 42); // should fail
 *
 * @param O object or symbol to be initialized
 */
void ESBMC_init_object(void *O);

// TODO: Implementations should be done at clang through defines (with versioning)


#ifdef __cplusplus
}
#endif
#endif // __ESBMC_H