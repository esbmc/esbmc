#pragma once

#include <goto-programs/goto_functions.h>
#include <util/symtab/context.h>

/// Hoist function calls that sit inside an expression out to statement level.
///
/// CBMC serialises intrinsics such as `object_size` as expressions, and
/// `migrate_expr` maps them to calls because resolving them needs symex. On the
/// C frontend path `goto_convertt::remove_sideeffects` would lift those; a
/// goto-binary never runs goto_convert, so without this the call reaches the
/// solver as a value. Each nested call becomes `tmp = call(...)` inserted
/// before its instruction, with the node replaced by `tmp`.
void lift_call_expressions(contextt &context, goto_functionst &goto_functions);
