#pragma once

#include <goto-programs/goto_program.h>
#include <irep2/irep2_expr.h>
#include <util/symtab/namespace.h>

/* Debug census for #7521 Phase 0 (--field-sensitivity-census). Classifies
 * every struct or union object symex sees; reverted once measured. */
namespace field_census
{
extern bool enabled;
extern const namespacet *ns;

void note_instruction(const goto_programt::instructiont &insn);

/* A dereference resolved to root_object. field_path is true when the access
 * used a constant offset and built no byte-level or reinterpreting expression
 * over it. */
void note_deref(const expr2tc &root_object, bool field_path);
bool reinterprets(const expr2tc &value);

void report();
} // namespace field_census
