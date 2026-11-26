#pragma once

#include <goto-programs/goto_functions.h>
#include <util/symtab/context.h>

/** Flatten main into a single loop when it has more than one loop, or a jump
 * enters its loop past the head, so that extraction sees one loop. Each cut
 * point (a loop head, or a jump target that enters a loop or the code after
 * the first loop head from outside) gets a value of a location variable; a
 * dispatcher at the new loop head jumps to the cut point the location names.
 *
 *     prefix                      prefix
 *   h1: loop 1                  d: if loc == 0 goto h1
 *     goto h1          =>         goto h2
 *   h2: loop 2                  h1: loop 1, back edge goto s1
 *     goto h2                     goto s2
 *     suffix                    h2: loop 2, back edge goto s2
 *                                 suffix; goto end
 *                               s1: loc = 0; goto latch
 *                               s2: loc = 1; goto latch
 *                               latch: goto d
 *                               end:
 *
 * The prefix enters through `loc = 0; goto d`. Variables declared in a block
 * need no hoisting: symex gives a local one level-1 name per frame, and the
 * havoc covers what the loop declares.
 */
void flatten_loops(goto_functionst &goto_functions, contextt &context);
