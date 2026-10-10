#pragma once

class goto_functionst;
class contextt;

/// Lower longjmp's non-local transfer of control to guarded gotos.
///
/// The setjmp model stores a fresh token in the jmp_buf, which each call site
/// copies into a frame-local slot. The longjmp model arms
/// __ESBMC_longjmp_{pending,token,value}. After every call, a pending longjmp
/// branches to its function's dispatch block, which resumes after the setjmp
/// site whose slot holds the token, with the longjmp value as setjmp's result,
/// or else returns so the caller dispatches in turn. A no-op for a program
/// that never calls longjmp.
void lower_longjmp(goto_functionst &goto_functions, contextt &context);
