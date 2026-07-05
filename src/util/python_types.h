#pragma once

#include <util/irep.h>

class typet;

typet none_type();

typet any_type();

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

// Stamp a Python internal-aggregate kind ("tuple", "dict", "optional") onto a
// freshly created struct type.
void set_python_aggregate_kind(typet &type, const irep_idt &kind);

// Return the stamped kind, or an empty id for user class instances / non-Python
// structs.
irep_idt python_aggregate_kind(const typet &type);

// True for tuple / dict / Optional model aggregates; false for user-defined
// Python class instances.
bool is_python_internal_aggregate(const typet &type);
