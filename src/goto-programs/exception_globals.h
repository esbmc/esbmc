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
///   $esbmc_exc_uncaught_count : size_t — number of exceptions thrown (or
///                                   rethrown) in this thread that have not yet
///                                   entered their matching handler; backs
///                                   std::uncaught_exception(s) ([except.uncaught]).
///   $esbmc_exc_terminate_reason : size_t — classification of the terminate
///                                   point that routed into std::terminate(),
///                                   so the default handler can keep the
///                                   original diagnostic (e.g. noexcept
///                                   violation vs uncaught exception).
///
/// The typeid is a size_t to line up with the Python frontend's existing
/// PyObject.type_id field and with exception_typeidt's unsigned ids; the
/// value is a void* the handler static_casts to the caught type.
namespace exception_globals
{
constexpr const char *thrown_id = "c:@__ESBMC_exc_thrown";
constexpr const char *typeid_id = "c:@__ESBMC_exc_typeid";
constexpr const char *value_id = "c:@__ESBMC_exc_value";
constexpr const char *uncaught_count_id = "c:@__ESBMC_exc_uncaught_count";
constexpr const char *terminate_reason_id = "c:@__ESBMC_exc_terminate_reason";
constexpr const char *push_handled_id = "c:@F@__ESBMC_push_handled_exception";
constexpr const char *pop_handled_id = "c:@F@__ESBMC_pop_handled_exception";
constexpr const char *rethrow_current_id =
  "c:@F@__ESBMC_rethrow_current_exception";

enum terminate_reasont : unsigned
{
  terminate_reason_generic = 0,
  terminate_reason_uncaught = 1,
  terminate_reason_noexcept = 2,
  terminate_reason_exception_spec = 3,
  terminate_reason_no_active = 4,
};
} // namespace exception_globals

/// Idempotently register the exception-state globals in @p context,
/// zero-initialised (thrown=false, typeid=0, value=NULL, uncaught_count=0). If a
/// symbol already exists — e.g. the operational-model std::uncaught_exceptions()
/// pulled in `__ESBMC_exc_uncaught_count` as an extern declaration during library
/// linking — its storage flags are upgraded in place (static, thread-local,
/// zero-initialised) rather than left as a bare declaration. Safe to call once
/// per run before lowering.
void create_exception_state_symbols(contextt &context);
