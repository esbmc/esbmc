#pragma once

#include <util/irep.h>

class contextt;

/// Global symbols that carry the in-flight exception across the GOTO program
/// once throw/catch dispatch is lowered to ordinary control flow (issue
/// #5075). Together they replace the imperative `stack_catch`/`thrown_obj_map`
/// state held in goto_symext: the lowering pass arms them at a `throw`, and
/// handler landing pads read them in guarded gotos.
///
///   $esbmc_exc_thrown  : _Bool   — is an exception currently propagating?
///   $esbmc_exc_typeid  : size_t  — dynamic type id of the thrown object,
///                                   from exception_typeidt (0 == none).
///   $esbmc_exc_value   : void*   — pointer to the most-derived thrown
///                                   object, so handlers can bind/slice it.
///
/// The typeid is a size_t to line up with the Python frontend's existing
/// PyObject.type_id field and with exception_typeidt's unsigned ids; the
/// value is a void* the handler static_casts to the caught type.
namespace exception_globals
{
constexpr const char *thrown_id = "c:@__ESBMC_exc_thrown";
constexpr const char *typeid_id = "c:@__ESBMC_exc_typeid";
constexpr const char *value_id = "c:@__ESBMC_exc_value";
} // namespace exception_globals

/// Idempotently register the three exception-state globals in @p context,
/// zero-initialised (thrown=false, typeid=0, value=NULL). A no-op if they are
/// already present, so it is safe to call once per run before lowering.
void create_exception_state_symbols(contextt &context);
