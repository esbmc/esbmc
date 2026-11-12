#pragma once

#include <goto-programs/goto_functions.h>
#include <util/symtab/context.h>

/// Gives each field of a struct variable that is only reached by name a
/// variable of its own, named `s.f`, so writing one field no longer makes a
/// new value of the whole struct (#7521). A variable whose address is taken, a
/// parameter, an extern, the result of a call, or one assigned an expression
/// that reads it keeps its struct. Whole-struct reads are rebuilt from the
/// fields. Does nothing when the program may create threads.
void scalarize_structs(goto_functionst &goto_functions, contextt &context);
