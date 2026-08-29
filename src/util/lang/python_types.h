#pragma once

#include <util/irep/irep.h>

class typet;

typet none_type();

typet any_type();

// True when a function yields nothing a postcondition could name: `-> None`,
// no annotation, or no enclosing function at all. Shared so the clause check
// and the __ESBMC_return_value lowering cannot disagree about what None means.
bool returns_no_value(const typet &t);

// Classification of Python "internal model aggregate" struct types.
//
// The Python frontend lowers a handful of built-in container/union types to
// plain structs whose representation and lifetime are managed by the
// operational model rather than by user code: tuples, dicts and typing
// Optional unions. These must be distinguished from user-defined Python class
// instances, which behave like ordinary garbage-collected objects.
//
// Historically the distinction was recovered by substring-matching the struct
// tag (e.g. tag.find("dict_") != npos, tag.rfind("tag-tuple", 0) == 0). That
// is brittle -- a user class named "dict_node" matches "dict_" -- and the
// literal tag spellings were duplicated across the frontend and goto-symex
// behind a "keep in sync" comment. Instead, each aggregate type is stamped
// with an explicit kind attribute at creation time and every consumer reads it
// back. The attribute lives on the (old-irep) struct type, survives
// symbol-table storage and namespacet::follow, and is read both by the Python
// frontend (dunder dispatch, list object-ref) and by goto-symex (object GC
// lifetime).
//
// The "#"-prefixed name routes to the irep comment slot, so the stamp does not
// perturb type identity (irept::operator== ignores comments) -- existing
// type-equality and symbol-matching logic is unaffected. The flip side is that
// the kind is preserved only across copies / follow: rebuilding a type from its
// tag and components drops it. All tuple/dict/Optional types are produced by
// the factory functions in this frontend (which both tag and stamp), and the
// consumers read the factory-produced type by copy, so this holds today; a new
// code path that reconstructs one of these structs by hand must re-stamp it.
#define PYTHON_AGGREGATE_ATTR "#python_aggregate"

// Marks the null void* the frontend returns for a method call it could not
// resolve to a class. The value is only a placeholder: reading it as a
// truthiness test would prove `not call()` from an inference gap, so the `not`
// arm replaces a tagged operand with a nondet bool. Every other use keeps the
// constant, whose folding is what stops method-heavy programs from blowing up.
#define PYTHON_UNRESOLVED_CALL_ATTR "#python_unresolved_call"

// Stamp a Python internal-aggregate kind ("tuple", "dict", "optional") onto a
// freshly created struct type.
void set_python_aggregate_kind(typet &type, const irep_idt &kind);

// Return the stamped kind, or an empty id for user class instances / non-Python
// structs.
irep_idt python_aggregate_kind(const typet &type);

// True for tuple / dict / Optional model aggregates; false for user-defined
// Python class instances.
bool is_python_internal_aggregate(const typet &type);
