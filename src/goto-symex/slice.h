#ifndef CPROVER_GOTO_SYMEX_SLICE_H
#define CPROVER_GOTO_SYMEX_SLICE_H

#include <goto-symex/symex_target_equation.h>

/**
 * Marks SSA_steps to be ignored which have no effects on the target equation,
 * according to the options set in the `config`.
 *
 * Notably, this function depends on the global `config`:
 *  - "no-slice" in `options` -> perform only simple slicing: ignore everything
 *    after the final assertion
 *  - "slice-assumes" in `options` -> also perform slicing of assumption steps
 *  - `config.no_slice_names` and `config.no_slice_ids` -> suppress slicing of
 *    particular symbols in non-simple slicing mode.
 *
 * @param eq The target equation containing the SSA steps to perform program
 *           slicing on.
 * @return The number of ignored SSA steps due to this slicing.
 */
BigInt slice(std::shared_ptr<symex_target_equationt> &eq);

#endif
