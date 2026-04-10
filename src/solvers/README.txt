This directory contains the ESBMC SMT reduction code. The primary purpose is
to reduce SSA programs that contain expressions over integers, arrays,
pointers, structures, complex byte operations, and much more, down to an SMT
formula.

The best reference is the boolector backend; see
solvers/boolector. There's also some documentation on how the new solver
backends are arranged in solvers/smt/smt_conv.h in a file comment. There
are also (mostly) comments about the purpose of each class in that file.
A brief overview:

There are three main abstract classes: smt_convt, smt_ast, and smt_sort.
Each solver backend subclasses these classes into their own, for
example, boolector_convt, btor_smt_ast, etc. (The boolector backend
doesn't subclass smt_sort as it's unnecessary). The 'conv' class is
supposed to represent an instance of the solver; the ast class holds
instances of pieces of SMT ast (functions, constants, whatever), and the
sort class holds instances of sorts.

Literals / constants are made with methods like 'mk_smt_bool',
'mk_smt_int' and so forth, as implemented by the convt class. They
return ast's. There's a similar mechanism for making sorts, mk_sort [0].

All the main work happens in the 'mk_func_app' method: it takes a
function kind, some ast arguments to operate on, and the
resulting sort. The solver 'convt' class takes the ast arguments,
creates the function application in the solver, wraps it in a new ast
object, and returns it. See boolector_convt::mk_func_app to see that
this is fairly simple: it casts smt_ast's to its own btor_smt_ast
class, then calls boolector_add, boolector_sub, etc, to create function
applications. Those are then wrapped via the new_ast method and returned.

The point of this interface is to avoid converting big, complicated,
possibly non-SMT expressions in the solver backend -- that's all handled
in the (abstract) smt_convt class. All the backend has to do is
implement the solver-specific parts of a general SMT solver, i.e.
something that constructs formulae out of function applications.

There's some funkyness to do with array's and tuples: most SMT solvers
don't support tuples, and I tried to keep the door open for SAT solvers
by abstracting arrays. There are two "interfaces" in
solvers/smt/smt_array.h and solvers/smt/smt_tuple.h, which solvers have
to implement if they support arrays or tuples; plus some virtual methods
in smt_ast. It's worth reading up on C++ virtual interface classes
before dealing with this (tl;dr, like java interfaces, but worse. It's
essentially passing a vtable around).

Finally, there's some boilerplate in solvers/solve.cpp for creating
solvers. The idea is to implement the factory pattern for solver
creation, avoiding general-esbmc code having to touch solver-specific stuff.

Interval/Real-Arithmetic (--ir-ieee) mode
------------------------------------------
When the --ir-ieee flag is enabled, floating-point operations are encoded
in real arithmetic rather than bit-precise FP. The key entry point is
smt_convt::apply_ieee754_semantics (smt_conv.cpp), which wraps each
real-valued FP result in a sound symmetric error enclosure.

For each FP operation whose exact real value is r, the round-to-nearest
rounding error satisfies the standard model:

  |fl(r) - r| <= eps_rel * |r| + eps_abs

where eps_rel is half the machine epsilon (2^-53 for double, 2^-24 for
single) and eps_abs is the minimum positive subnormal (2^-1074 for
double, 2^-149 for single), covering the underflow region. The enclosure
is therefore:

  ra_lo = r - (eps_rel * |r| + eps_abs)
  ra_hi = r + (eps_rel * |r| + eps_abs)

and the assertions ra_lo <= result <= ra_hi and ra_lo <= ra_hi are
added to the formula. The enclosure bounds are pinned via bidirectional
inequalities (rather than equalities) to survive Z3's solve-eqs tactic.

The epsilon constants are provided by four helpers in smt_conv.cpp:
get_double_eps_rel, get_single_eps_rel, get_double_min_subnormal, and
get_single_min_subnormal. Each decimal literal is rounded upward at the
last digit to guarantee the SMT solver parses it as a value >= the true
power-of-two, preserving soundness of the enclosure.

Non-standard FP formats fall back to an unconstrained (weak) enclosure
until theorem-driven bounds are implemented for them.

[0] It's variardic; I thought that'd be useful then; see
boolector_convt::mk_sort for an implementation. In reality, it could have
been more useful.
