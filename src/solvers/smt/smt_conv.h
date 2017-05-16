#ifndef _ESBMC_PROP_SMT_SMT_CONV_H_
#define _ESBMC_PROP_SMT_SMT_CONV_H_

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index_container.hpp>
#include <cstdint>
#include <solvers/prop/literal.h>
#include <solvers/prop/pointer_logic.h>
#include <util/irep2.h>
#include <util/message.h>
#include <util/namespace.h>
#include <util/threeval.h>
#include <util/type_byte_size.h>

/** @file smt_conv.h
 *  SMT conversion tools and utilities.
 *  smt_convt is the base class for everything that attempts to convert the
 *  contents of an SSA program into something else, generally SMT or SAT based.
 *
 *  The class itself does various accounting and structuring of the conversion,
 *  however the challenge is that as we convert the SSA program into anything
 *  else, we must deal with the fact that expressions in ESBMC are somewhat
 *  bespoke, and don't follow any particular formalism or logic. Therefore
 *  a lot of translation has to occur to reduce it to the desired logic, a
 *  process that Kroening refers to in CBMC as 'Flattenning'.
 *
 *  The conceptual data flow is that an SSA program held by
 *  symex_target_equationt is converted into a series of boolean propositions
 *  in some kind of solver context, the handle to which are objects of class
 *  smt_ast. These are then asserted as appropriate (or conjoined or
 *  disjuncted), after which the solver may be asked whether the formula is
 *  or not. If it is, the value of symbols in the formula may be retrieved
 *  from the solver.
 *
 *  To do that, the user must allocate a solver converter object, which extends
 *  the class smt_convt. Current, create_solver_factory will do this, in the
 *  factory-pattern manner (ish). Each solver converter implements all the
 *  abstract methods of smt_convt. When handed an expression to convert,
 *  smt_convt deconstructs it into a series of function applications, which it
 *  creates by calling various abstract methods implemented by the converter
 *  (in particular mk_func_app).
 *
 *  The actual function applications are in smt_ast objects. Following the
 *  SMTLIB definition, these are basically a term.
 *
 *  In no particular order, the following expression translation problems exist
 *  and are solved at various layers:
 *
 *  For all solvers, the following problems are flattenned:
 *    * The C memory address space
 *    * Representation of pointer types
 *    * Casts
 *    * Byte operations on objects (extract/update)
 *    * FixedBV representation of floats
 *    * Unions -> something else
 *
 *  While these problems are supported by some SMT solvers, but are flattened
 *  in others (as SMT doesn't support these):
 *    * Bitvector integer overflow detection
 *    * Tuple representation (and arrays of them)
 *
 *  SAT solvers have the following aspects flattened:
 *    * Arrays (using Kroenings array decision procedure)
 *    * First order logic bitvector calculations to boolean formulas
 *    * Boolean formulas to CNF
 *
 *  If you find yourself having to make the SMT translation translate more than
 *  these things, ask yourself whether what you're doing is better handled at
 *  a different layer, such as symbolic execution. A nonexhaustive list of these
 *  include:
 *    * Anything involving pointer dereferencing at all
 *    * Anything that considers the control flow guard at any point
 *    * Pointer liveness or dynamic allocation consideration
 *
 *  A reasonable amount of SMT conversion complexity comes from the
 *  eccentricities of different solvers. This is a necessary part of working in
 *  the tar pit.
 *
 *  @see smt_convt
 *  @see symex_target_equationt
 *  @see create_solver_factory
 *  @see smt_convt::mk_func_app
 */

class smt_convt; // Forward dec.

/** Identifier for SMT sort kinds
 *  Each different kind of sort (i.e. arrays, bv's, bools, etc) gets its own
 *  identifier. To be able to describe multiple kinds at the same time, they
 *  take binary values, so that they can be used as bits in an integer. */
enum smt_sort_kind {
  SMT_SORT_INT = 1,
  SMT_SORT_REAL = 2,
  SMT_SORT_BV = 4,
  SMT_SORT_ARRAY = 8,
  SMT_SORT_BOOL = 16,
  SMT_SORT_STRUCT = 32,
  SMT_SORT_UNION = 64, // Contencious
  SMT_SORT_FLOATBV = 128,
  SMT_SORT_FLOATBV_RM = 256
};

/** Identifiers for SMT functions.
 *  Each SMT function gets a unique identifier, representing its interpretation
 *  when applied to some arguments. This can be used to describe a function
 *  application when joined with some arguments. Initial values such as
 *  terminal functions (i.e. bool, int, symbol literals) shouldn't normally
 *  be encountered and instead converted to an smt_ast before use. The
 *  'HACKS' function represents some kind of special case, according to where
 *  it is encountered; the same for 'INVALID'.
 *
 *  @see smt_convt::convert_terminal
 *  @see smt_convt::convert_ast
 */
enum smt_func_kind {
  // Terminals
  SMT_FUNC_INVALID = 0, // For conversion lookup table only
  SMT_FUNC_HACKS = 1, // indicate the solver /has/ to use the temp expr.
  SMT_FUNC_INT = 2,
  SMT_FUNC_BOOL,
  SMT_FUNC_BVINT,
  SMT_FUNC_REAL,
  SMT_FUNC_SYMBOL,

  // Nonterminals
  SMT_FUNC_ADD,
  SMT_FUNC_BVADD,
  SMT_FUNC_SUB,
  SMT_FUNC_BVSUB,
  SMT_FUNC_MUL,
  SMT_FUNC_BVMUL,
  SMT_FUNC_DIV,
  SMT_FUNC_BVUDIV,
  SMT_FUNC_BVSDIV,
  SMT_FUNC_MOD,
  SMT_FUNC_BVSMOD,
  SMT_FUNC_BVUMOD,
  SMT_FUNC_SHL,
  SMT_FUNC_BVSHL,
  SMT_FUNC_BVASHR,
  SMT_FUNC_NEG,
  SMT_FUNC_BVNEG,
  SMT_FUNC_BVLSHR,
  SMT_FUNC_BVNOT,
  SMT_FUNC_BVNXOR,
  SMT_FUNC_BVNOR,
  SMT_FUNC_BVNAND,
  SMT_FUNC_BVXOR,
  SMT_FUNC_BVOR,
  SMT_FUNC_BVAND,

  // Logic
  SMT_FUNC_IMPLIES,
  SMT_FUNC_XOR,
  SMT_FUNC_OR,
  SMT_FUNC_AND,
  SMT_FUNC_NOT,

  // Comparisons
  SMT_FUNC_LT,
  SMT_FUNC_BVSLT,
  SMT_FUNC_BVULT,
  SMT_FUNC_GT,
  SMT_FUNC_BVSGT,
  SMT_FUNC_BVUGT,
  SMT_FUNC_LTE,
  SMT_FUNC_BVSLTE,
  SMT_FUNC_BVULTE,
  SMT_FUNC_GTE,
  SMT_FUNC_BVSGTE,
  SMT_FUNC_BVUGTE,

  SMT_FUNC_EQ,
  SMT_FUNC_NOTEQ,

  SMT_FUNC_ITE,

  SMT_FUNC_STORE,
  SMT_FUNC_SELECT,

  SMT_FUNC_CONCAT,
  SMT_FUNC_EXTRACT, // Not for going through mk app due to sillyness.

  SMT_FUNC_INT2REAL,
  SMT_FUNC_REAL2INT,
  SMT_FUNC_IS_INT,

  // floatbv operations
  SMT_FUNC_FABS,
  SMT_FUNC_ISZERO,
  SMT_FUNC_ISNAN,
  SMT_FUNC_ISINF,
  SMT_FUNC_ISNORMAL,
  SMT_FUNC_ISNEG,
  SMT_FUNC_ISPOS,
  SMT_FUNC_IEEE_EQ,
  SMT_FUNC_IEEE_ADD,
  SMT_FUNC_IEEE_SUB,
  SMT_FUNC_IEEE_MUL,
  SMT_FUNC_IEEE_DIV,
  SMT_FUNC_IEEE_FMA,

  SMT_FUNC_IEEE_RM_NE,
  SMT_FUNC_IEEE_RM_ZR,
  SMT_FUNC_IEEE_RM_PI,
  SMT_FUNC_IEEE_RM_MI,

  SMT_FUNC_BV2FLOAT,
  SMT_FUNC_FLOAT2BV,
};

/** Class that will hold information about which operation
 *  will be applied for a particular type
 */
struct expr_op_convert {
  smt_func_kind int_encoding_func;
  smt_func_kind signedbv_func;
  smt_func_kind unsignedbv_func;
  smt_func_kind fixedbv_func;
  smt_func_kind floatbv_func;
};

/** A class for storing an SMT sort.
 *  This class abstractly represents an SMT sort: solver converter classes are
 *  expected to extend this and add fields that store their solvers
 *  representation of the sort. Then, this base class is used as a handle
 *  through the rest of the SMT conversion code.
 *
 *  Only a few piece of sort information are used to make conversion decisions,
 *  and are thus actually stored in the sort object itself.
 *  @see smt_ast
 */

class smt_sort;
typedef const smt_sort * smt_sortt;

class smt_sort {
public:
  /** Identifies what /kind/ of sort this is.
   *  The specific sort itself may be parameterised with widths and domains,
   *  for example. */
  smt_sort_kind id;
  /** Data size of the sort.
   *  For bitvectors this is the bit width, for arrays the range BV bit width.
   *  For everything else, undefined */
  unsigned long data_width;
  /** BV Width of array domain. For everything else, undefined */
  unsigned long domain_width;

  smt_sort(smt_sort_kind i) : id(i), data_width(0), domain_width(0)
  {
    assert(id != SMT_SORT_ARRAY);
  }
  smt_sort(smt_sort_kind i, unsigned long width)
    : id(i), data_width(width), domain_width(0)
  {
    assert(width != 0 || i == SMT_SORT_INT);
    assert(id != SMT_SORT_ARRAY);
  }
  smt_sort(smt_sort_kind i, unsigned long rwidth, unsigned long domwidth)
    : id(i), data_width(rwidth), domain_width(domwidth) {
      assert(id == SMT_SORT_ARRAY);
      //assert(domain_width != 0);
      // XXX not applicable during int mode?
    }

  virtual ~smt_sort() { }
};

#define is_tuple_ast_type(x) (is_structure_type(x) || is_pointer_type(x))

inline bool is_tuple_array_ast_type(const type2tc &t)
{
  if (!is_array_type(t))
    return false;

  const array_type2t &arr_type = to_array_type(t);
  type2tc range = arr_type.subtype;
  while (is_array_type(range))
    range = to_array_type(range).subtype;

  return is_tuple_ast_type(range);
}

/** Storage of an SMT function application.
 *  This class represents a single SMT function app, abstractly. Solver
 *  converter classes must extend this and add whatever fields are necessary
 *  to represent a function application in the solver they support. A converted
 *  expression becomes an SMT function application; that is then handed around
 *  the rest of the SMT conversion code as an smt_ast.
 *
 *  While an expression becomes an smt_ast, the inverse is not true, and a
 *  single expression may in fact become many smt_asts in various places. See
 *  smt_convt for more detail on how conversion occurs.
 *
 *  The function arguments, and the actual function application itself are all
 *  abstract and dealt with by the solver converter class. Only the sort needs
 *  to be available for us to make conversion decisions.
 *  @see smt_convt
 *  @see smt_sort
 */

class smt_ast;
typedef const smt_ast * smt_astt;

class smt_ast {
public:
  /** The sort of this function application. */
  smt_sortt sort;

  smt_ast(smt_convt *ctx, smt_sortt s);
  virtual ~smt_ast() { }

  // "this" is the true operand.
  virtual smt_astt ite(smt_convt *ctx, smt_astt cond,
      smt_astt falseop) const;

  /** Abstractly produce an equality. Does the right thing (TM) whether it's
   *  a normal piece of AST or a tuple / array.
   *  @param ctx SMT context to produce the equality in.
   *  @param other Piece of AST to compare 'this' with.
   *  @return Boolean typed AST representing an equality */
  virtual smt_astt eq(smt_convt *ctx, smt_astt other) const;

  /** Abstractly produce an assign. Defaults to being an equality, however
   *  for some special cases up to the backend, there may be optimisations made
   *  for array or tuple assigns, and so forth.
   *  @param ctx SMT context to do the assignment in.
   *  @param sym Symbol to assign to
   *  @return AST representing the assigned symbol */
  virtual void assign(smt_convt *ctx, smt_astt sym) const;

  /** Abstractly produce an "update", i.e. an array 'with' or tuple 'with'.
   *  @param ctx SMT context to make this update in.
   *  @param value Value to insert into the updated field
   *  @param idx Array index or tuple field
   *  @param idx_expr If an array, expression representing the index
   *  @return AST of this' type, representing the update */
  virtual smt_astt update(smt_convt *ctx, smt_astt value,
                                unsigned int idx,
                                expr2tc idx_expr = expr2tc()) const;

  /** Select a value from an array, for both normal arrays and tuple arrays.
   *  @param ctx SMT context to produce this in.
   *  @param idx Index to select the value from.
   *  @return AST of the array's range sort representing the selected item */
  virtual smt_astt select(smt_convt *ctx, const expr2tc &idx) const;

  /** Project a member from a structure, or an field-array from a struct array.
   *  @param ctx SMT context to produce this in.
   *  @param elem Struct index to project.
   *  @return AST representing the chosen element / element-array */
  virtual smt_astt project(smt_convt *ctx, unsigned int elem) const;

  virtual void dump() const = 0;
};

// Pull in the tuple interface definitions. _after_ the AST defs.
#include <solvers/smt/smt_tuple.h>

// Also, array interface
#include <solvers/smt/smt_array.h>

/** The base SMT-conversion class/interface.
 *  smt_convt handles a number of decisions that must be made when
 *  deconstructing ESBMC expressions down into SMT representation. See
 *  smt_conv.h for more high level documentation of this.
 *
 *  The basic flow is thus: a class that can create SMT formula in some solver
 *  subclasses smt_convt, implementing abstract methods, in particular
 *  mk_func_app. The rest of ESBMC then calls convert with an expression, and
 *  this class deconstructs it into a series of applications, as documented by
 *  the smt_func_kind enumeration. These are then created via mk_func_app or
 *  some more specific method calls. Boolean sorted ASTs are then asserted
 *  into the solver context.
 *
 *  The exact lifetime of smt asts here is currently undefined, unfortunately,
 *  although smt_convt posesses a cache, so they generally have a reference
 *  in there. This will probably be fixed in the future.
 *
 *  In theory this class supports pushing and popping of solver contexts,
 *  although of course that depends too on the subclass supporting it. However,
 *  this hasn't really been tested since everything here was rewritten from
 *  The Old Way, so don't trust it.
 *
 *  While mk_func_app is supposed to be the primary interface to making SMT
 *  function applications, in some cases we want to introduce some
 *  abstractions, and this becomes unweildy. Thus, tuple and array operations
 *  are performed via virtual function calls. By default, array operations are
 *  then passed through to mk_func_app, while tuples are decomposed into sets
 *  of variables which are then created through mk_func_app. If this isn't
 *  what a solver wants to happen, it can override this and handle that itself.
 *  The idea is that, in the manner of metaSMT, you can then compose a series
 *  of subclasses that perform the desired amount of flattening, and then work
 *  from there. (Some of this is still WIP though).
 *
 *  NB: the whole smt_asts-are-const situation needs to be resolved too.
 *
 *  @see smt_conv.h
 *  @see smt_func_kind */
class smt_convt : public messaget
{
public:
  /** Shorthand for a vector of smt_ast's */
  typedef std::vector<smt_astt > ast_vec;

  /** Primary constructor. After construction, smt_post_init must be called
   *  before the object is used as a solver converter.
   *
   *  @param int_encoding Whether nor not we should use QF_AUFLIRA or QF_AUFBV.
   *  @param _ns Namespace for looking up the type of certain symbols. */
  smt_convt(bool int_encoding, const namespacet &_ns);
  ~smt_convt();

  /** Post-constructor setup method. We must create various pieces of memory
   *  model data for tracking, however can't do it from the constructor because
   *  the solver converter itself won't have been initialized itself at that
   *  point. So, once it's ready, the solver converter should call this from
   *  it's constructor. */
  void smt_post_init(void);

  // The API that we provide to the rest of the world:
  /** @{
   *  @name External API to smt_convt. */

  /** Result of a call to dec_solve. Either sat, unsat, or error. SMTLIB is
   *  historic case that needs to go. */
  typedef enum { P_SATISFIABLE, P_UNSATISFIABLE, P_ERROR, P_SMTLIB } resultt;

  /** Push one context on the SMT assertion stack. */
  virtual void push_ctx(void);
  /** Pop one context on the SMT assertion stack. */
  virtual void pop_ctx(void);

  /** Main interface to SMT conversion.
   *  Takes one expression, and converts it into the underlying SMT solver,
   *  returning a single smt_ast that represents the converted expressions
   *  value. The lifetime of the returned pointer is currently undefined.
   *
   *  @param expr The expression to convert into the SMT solver
   *  @return The resulting handle to the SMT value. */
  smt_astt convert_ast(const expr2tc &expr);

  /** Interface to specifig SMT conversion.
   *  Takes one expression, and converts it into the underlying SMT solver,
   *  depending on the type of the expression.
   *
   *  @param expr The expression to convert into the SMT solver
   *  @param type The expression's type
   *  @param args The expression's args
   *  @param ops The operations for each sort type
   *  @return The resulting handle to the SMT value. */
  smt_astt convert_ast(
    const expr2tc &expr,
    const type2tc &type,
    smt_astt const *args,
    struct expr_op_convert ops);

  void convert_assign(const expr2tc &expr);

  /** Make an n-ary 'or' function application.
   *  Takes a vector of smt_ast's, all boolean sorted, and creates a single
   *  'or' function app over all the smt_ast's.
   *  @param v The vector of converted boolean expressions to be 'or''d.
   *  @return The smt_ast handle to the 'or' func app. */
  virtual smt_astt make_disjunct(const ast_vec &v);

  /** Make an n-ary 'and' function application.
   *  Takes a vector of smt_ast's, all boolean sorted, and creates a single
   *  'and' function app over all the smt_ast's.
   *  @param v The vector of converted boolean expressions to be 'and''d.
   *  @return The smt_ast handle to the 'and' func app. */
  virtual smt_astt make_conjunct(const ast_vec &v);

  /** Create the inverse of an smt_ast. Equivalent to a 'not' operation.
   *  @param a The ast to invert. Must be boolean sorted.
   *  @return The inverted piece of AST. */
  smt_astt invert_ast(smt_astt a);

  /** Create an ipmlication between two smt_ast's.
   *  @param a The ast that implies the truth of the other. Boolean.
   *  @param b The ast whos truth is implied. Boolean.
   *  @return The resulting piece of AST. */
  smt_astt imply_ast(smt_astt a, smt_astt b);

  /** Assert the truth of an ast. Equivalent to the 'assert' directive in the
   *  SMTLIB language, this informs the solver that in the satisfying
   *  assignment it attempts to produce, the formula corresponding to the
   *  smt_ast argument must evaluate to true.
   *  @param a A handle to the formula that must be true. */
  virtual void assert_ast(smt_astt a) = 0;

  /** Solve the formula given to the solver. The solver will attempt to produce
   *  a satisfying assignment for all of the variables / symbols used in the
   *  formula, where all the asserted sub-formula are true. Results are either
   *  unsat (the formula is inconsistent), sat (an assignment exists), or that
   *  an error occurred.
   *  @return Result code of the call to the solver. */
  virtual resultt dec_solve() = 0;

  void pre_solve();

  /** Fetch a satisfying assignment from the solver. If a previous call to
   *  dec_solve returned satisfiable, then the solver has a set of assignments
   *  to symbols / variables used in the formula. This method retrieves the
   *  value of a symbol, and formats it into an ESBMC expression.
   *  @param expr Variable to get the value of. Must be a symbol expression.
   *  @return Explicit assigned value of expr in the solver. May be nil, in
   *          which case the solver did not assign a value to it for some
   *          reason. */
  virtual expr2tc get(const expr2tc &expr);

  /** Solver name fetcher. Returns a string naming the solver being used, and
   *  potentially it's version, if available.
   *  @return The name of the solver this smt_convt uses. */
  virtual const std::string solver_text()=0;

  /** Fetch the value of a boolean sorted smt_ast. (The 'l' is for literal, and
   *  is historic). Returns a three valued result, of true, false, or
   *  unassigned.
   *  @param a The boolean sorted ast to fetch the value of.
   *  @return A three-valued return val, of the assignment to a. */
  virtual tvt l_get(smt_astt a)=0;

  /** @} */

  /** @{
   *  @name Internal conversion API between smt_convt and solver converter */

  /** Create an SMT function application. Using the provided information,
   *  the solver converter should create a function application in the solver
   *  being used, then wrap it in an smt_ast, and return it. If the desired
   *  function application is not supported by the solver, print an error and
   *  abort.
   *
   *  @param s The resulting sort of the func app we are creating.
   *  @param k The kind of function application to create.
   *  @param args Array of function apps to use as arguments to this one.
   *  @param numargs The number of elements in args. Should be consistent with
   *         the function kind k.
   *  @return The resulting function application, wrapped in an smt_ast. */
  virtual smt_astt mk_func_app(smt_sortt s, smt_func_kind k,
                               smt_astt  const *args,
                               unsigned int numargs) = 0;

  // Some helpers

  inline smt_astt mk_func_app(smt_sortt s, smt_func_kind k, smt_astt arg1)
  {
    smt_astt args[1];
    args[0] = arg1;
    return mk_func_app(s, k, args, 1);
  }

  inline smt_astt mk_func_app(smt_sortt s, smt_func_kind k, smt_astt arg1,
      smt_astt arg2)
  {
    smt_astt args[2];
    args[0] = arg1;
    args[1] = arg2;
    return mk_func_app(s, k, args, 2);
  }

  inline smt_astt mk_func_app(smt_sortt s, smt_func_kind k, smt_astt arg1,
      smt_astt arg2, smt_astt arg3)
  {
    smt_astt args[3];
    args[0] = arg1;
    args[1] = arg2;
    args[2] = arg3;
    return mk_func_app(s, k, args, 3);
  }

  inline smt_astt mk_func_app(smt_sortt s, smt_func_kind k, smt_astt arg1,
      smt_astt arg2, smt_astt arg3, smt_astt arg4)
  {
    smt_astt args[4];
    args[0] = arg1;
    args[1] = arg2;
    args[2] = arg3;
    args[3] = arg4;
    return mk_func_app(s, k, args, 4);
  }

  /** Create an SMT sort. The sort kind k indicates what kind of sort to create,
   *  and the parameters of the sort are passed in as varargs. Briefly, these
   *  arguments are:
   *  * Bools: None
   *  * Int's: None
   *  * BV's:  Width as a machine integer, and a bool that's true if it's signed
   *  * Arrays: Two pointers to smt_sort's: the domain sort, and the range sort
   *
   *  Structs and unions use @ref mk_struct_sort and @ref mk_union_sort.
   *
   *  @param k The kind of SMT sort that will be created.
   *  @return The smt_sort wrapper for the sort. Lifetime currently undefined */
  virtual smt_sortt mk_sort(const smt_sort_kind k, ...) = 0;

  /** Create an integer smt_ast. That is, an integer in QF_AUFLIRA, rather than
   *  a bitvector.
   *  @param theint BigInt representation of the number to create.
   *  @param sign Whether this integer is considered signed or not.
   *  @return The newly created terminal smt_ast of this integer. */
  virtual smt_astt mk_smt_int(const mp_integer &theint, bool sign) = 0;

  /** Create a real in a smt_ast.
   *  @param str String representation of the real, to be parsed by the solver.
   *         Tends to be one integer divided ('/') by another. After inspecting
   *         all other options, there are none that are good, this is a
   *         legitimate use of strings.
   *  @return The newly created terminal smt_ast of this real. */
  virtual smt_astt mk_smt_real(const std::string &str) = 0;

  /** Create a bitvector.
   *  @param theint Integer representation of the bitvector. Any excess bits
   *         in the stored integer should be ignored.
   *  @param sign Whether this bitvector is to be considered signed or not.
   *  @param w Width, in bits, of the bitvector to create.
   *  @return The newly created terminal smt_ast of this bitvector. */
  virtual smt_astt mk_smt_bvint(const mp_integer &theint, bool sign,
                                unsigned int w) = 0;

  /** Create a floating point bitvector
   *  @param thereal the ieee float number
   *  @param ew Exponent width, in bits, of the bitvector to create.
   *  @param sw Significand width, in bits, of the bitvector to create.
   *  @return The newly created terminal smt_ast of this bitvector. */
  virtual smt_astt mk_smt_bvfloat(const ieee_floatt &thereal,
                                  unsigned ew, unsigned sw) = 0;

  /** Create a NaN floating point bitvector
   *  @param ew Exponent width, in bits, of the bitvector to create.
   *  @param sw Significand width, in bits, of the bitvector to create.
   *  @return The newly created terminal smt_ast of this bitvector. */
  virtual smt_astt mk_smt_bvfloat_nan(unsigned ew, unsigned sw) = 0;

  /** Create a (+/-)inf floating point bitvector
   *  @param sgn Whether this bitvector is negative or positive.
   *  @param ew Exponent width, in bits, of the bitvector to create.
   *  @param sw Significand width, in bits, of the bitvector to create.
   *  @return The newly created terminal smt_ast of this bitvector. */
  virtual smt_astt mk_smt_bvfloat_inf(bool sgn, unsigned ew, unsigned sw) = 0;

  /** Create a rounding mode to be used by floating point cast and arith ops
   *  @param rm the kind of rounding mode
   *  @return The newly created rounding mode smt_ast. */
  virtual smt_astt mk_smt_bvfloat_rm(ieee_floatt::rounding_modet rm) = 0;

  /** Typecast from a floating point
   *  @param cast the cast expression
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_typecast_from_bvfloat(const typecast2t &cast) = 0;

  /** Typecast to a floating point
   *  @param cast the cast expression
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_typecast_to_bvfloat(const typecast2t &cast) = 0;

  /** Calculate the nearby int from a floating point, considering the rounding mode
   *  @param expr the nearby int expression
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_nearbyint_from_float(const nearbyint2t &expr) = 0;

  /** Convert the ieee arithmetic operations (add, sub, mul, div, mod)
   *  @param expr the arithmetic operations
   *  @return The newly created cast smt_ast. */
  virtual smt_astt mk_smt_bvfloat_arith_ops(const expr2tc &expr) = 0;

  /** Create a boolean.
   *  @param val Whether to create a true or false boolean.
   *  @return The newly created terminal smt_ast of this boolean. */
  virtual smt_astt mk_smt_bool(bool val) = 0;

  /** Create a symbol / variable. These correspond to renamed SSA variables in
   *  the SSA program, although any other names can be used too, so long as they
   *  don't conflict with anything else.
   *  @param name Textual name of the symbol to create.
   *  @param s The sort of the symbol we're creating.
   *  @param The newly created terminal smt_ast of this symbol. */
  virtual smt_astt mk_smt_symbol(const std::string &name, smt_sortt s) =0;

    /** Create an 'extract' func app. Due to the fact that we can't currently
   *  encode integer constants as function arguments without serious faff,
   *  this can't be performed via the medium of mk_func_app. Hence, this api
   *  call.
   *  @param a The source piece of ast to extract a value from.
   *  @param high The topmost bit to select from the source, down to low.
   *  @param low The lowest bit to select from the source.
   *  @param s The sort of the resulting piece of ast. */
  virtual smt_astt mk_extract(smt_astt a, unsigned int high,
                              unsigned int low, smt_sortt s) = 0;

  /** Extract the assignment to a boolean variable from the SMT solvers model.
   *  @param a The AST whos value we wish to know.
   *  @return Expression representation of a's value, as a constant_bool2tc */
  virtual expr2tc get_bool(smt_astt a) = 0;

  /** Extract the assignment to a bitvector from the SMT solvers model.
   *  @param a The AST whos value we wish to know.
   *  @return Expression representation of a's value, as a constant_int2tc */
  virtual expr2tc get_bv(const type2tc &t, smt_astt a) = 0;

  /** @} */

  /** @{
   *  @name Integer overflow solver-converter API. */

  /** Detect integer arithmetic overflows. Takes an expression, that is one of
   *  add / sub / mul, and evaluates whether its operation applied to its
   *  operands will result in an integer overflow or underflow.
   *  @param expr Expression to test for arithmetic overflows in.
   *  @return Boolean valued AST representing whether an overflow occurs. */
  virtual smt_astt overflow_arith(const expr2tc &expr);

  /** Detect integer overflows in a cast. Takes a typecast2tc as an argument,
   *  and if it causes a decrease in integer width, then encodes a test that
   *  the dropped bits are never significant / used.
   *  @param expr Cast to test for dropped / overflowed data in.
   *  @return Boolean valued AST representing whether an overflow occurs. */
  virtual smt_astt overflow_cast(const expr2tc &expr);

  /** Detects integer overflows in negation. This only tests for the case where
   *  MIN_INT is being negated, in which case there is no positive
   *  representation of that number, and an overflow occurs. Evaluates to true
   *  if that can occur in the operand.
   *  @param expr A neg2tc to test for overflows in.
   *  @return Boolean valued AST representing whether an overflow occurs. */
  virtual smt_astt overflow_neg(const expr2tc &expr);

  /** Method to dump the SMT formula */
  virtual void dump_smt();

  /** @} */

  /** @{
   *  @name Array operations solver-converter API. */

  /** High level index expression conversion. Deals with several annoying
   *  corner cases that must be addressed, such as flattening multidimensional
   *  arrays into one domain sort, or turning bool arrays into bit arrays.
   *  XXX, why is this virtual?
   *  @param expr An index2tc expression to convert to an SMT AST.
   *  @return An AST representing the index operation in the expression. */
  virtual smt_astt convert_array_index(const expr2tc &expr);

  /** Partner method to convert_array_index, for stores.
   *  XXX, why is this virtual?
   *  @param expr with2tc operation to convert to SMT.
   *  @return AST representing the result of evaluating expr. */
  virtual smt_astt convert_array_store(const expr2tc &expr);

  /** @} */

  /** @{
   *  @name Internal foo. */

  /** Convert expression and assert that it is true or false, according to the
   *  value argument */
  virtual void set_to(const expr2tc &expr, bool value);

  /** Create a free variable with the given sort, and a unique name, with the
   *  prefix given in 'tag' */
  virtual smt_astt mk_fresh(smt_sortt s, const std::string &tag,
                            smt_sortt st = NULL);
  /** Create a previously un-used variable name with the prefix given in tag */
  std::string mk_fresh_name(const std::string &tag);

  void renumber_symbol_address(const expr2tc &guard,
                               const expr2tc &addr_symbol,
                               const expr2tc &new_size);

  /** Convert a type2tc into an smt_sort. This despatches control to the
   *  appropriate method in the subclassing solver converter for type
   *  conversion */
  smt_sortt convert_sort(const type2tc &type);
  /** Convert a terminal expression into an SMT AST. This despatches control to
   *  the appropriate method in the subclassing solver converter for terminal
   *  conversion */
  smt_astt convert_terminal(const expr2tc &expr);

  /** Flatten pointer arithmetic. When faced with an addition or subtraction
   *  between a pointer and some integer or other pointer, perform whatever
   *  multiplications or casting is requried to honour the C semantics of
   *  pointer arith. */
  smt_astt convert_pointer_arith(const expr2tc &expr, const type2tc &t);
  /** Compare two pointers. This attempts to optimise cases where we can avoid
   *  comparing the integer representation of a pointer, as that's hugely
   *  inefficient sometimes (and gets bitblasted).
   *  @param expr First pointer to compare
   *  @param expr2 Second pointer to compare
   *  @param templ_expr The comparision expression -- this method will look at
   *         the kind of comparison being performed, and make an appropriate
   *         decision.
   *  @return Boolean valued AST as appropriate to the requested comparision */
  smt_astt convert_ptr_cmp(const expr2tc &expr, const expr2tc &expr2,
                                 const expr2tc &templ_expr);
  /** Take the address of some kind of expression. This will abort if the given
   *  expression isn't based on some symbol in some way. (i.e., you can't take
   *  the address of an addition, but you can take the address of a member of
   *  a struct, for example). */
  smt_astt convert_addr_of(const expr2tc &expr);
  /** Handle union/struct based corner cases for member2tc expressions */
  smt_astt convert_member(const expr2tc &expr);
  /** Convert an identifier to a pointer. When given the name of a variable
   *  that we want to take the address of, this inspects our current tracking
   *  of addresses / variables, and returns a pointer for the given symbol.
   *  If it hasn't had its address taken before, performs any computations or
   *  address space juggling required to make a new pointer.
   *  @param expr The symbol2tc expression of this symbol.
   *  @param sym The textual representation of this symbol.
   *  @return A pointer-typed AST representing the address of this symbol. */
  smt_astt convert_identifier_pointer(const expr2tc &expr,
                                            std::string sym);

  smt_astt init_pointer_obj(unsigned int obj_num, const expr2tc &size);

  /** Given a signed, upwards cast, extends the sign of the given AST to the
   *  desired length.
   *  @param a The bitvector to upcast.
   *  @param s The resulting sort of this extension operation
   *  @param topbit The highest bit of the bitvector (1-based)
   *  @param topwidth The number of bits to extend the input by
   *  @return A bitvector with topwidth more bits, of the appropriate sign. */
  smt_astt convert_sign_ext(smt_astt a, smt_sortt s,
                                  unsigned int topbit, unsigned int topwidth);
  /** Identical to convert_sign_ext, but extends AST with zeros */
  smt_astt convert_zero_ext(smt_astt a, smt_sortt s,
                                  unsigned int topwidth);
  /** Checks for equality with NaN representation. */
  smt_astt convert_is_nan(const expr2tc &expr);
  /** Checks for equality with inf representation. */
  smt_astt convert_is_inf(const expr2tc &expr);
  /** Checks for equality with normal representation. */
  smt_astt convert_is_normal(const expr2tc &expr);
  /** Checks for equality with finite representation. */
  smt_astt convert_is_finite(const expr2tc &expr);
  /** Converts signbit representation. */
  smt_astt convert_signbit(const expr2tc &expr);
  /** Converts rounding mode for ieee fp operations. */
  smt_astt convert_rounding_mode(const expr2tc& expr);
  /** Converts equality between two floatbvs. */
  smt_astt convert_ieee_equal(const expr2tc &expr);
  /** Convert a byte_extract2tc, pulling a byte from the byte representation
   *  of some piece of data. */
  smt_astt convert_byte_extract(const expr2tc &expr);
  /** Convert a byte_update2tc, inserting a byte into the byte representation
   *  of some piece of data. */
  smt_astt convert_byte_update(const expr2tc &expr);
  /** Convert the given expr to AST, then assert that AST */
  void assert_expr(const expr2tc &e);
  /** Convert constant_array2tc's and constant_array_of2tc's */
  smt_astt array_create(const expr2tc &expr);
  /** Mangle constant_array / array_of data with tuple array type, into a
   *  more convenient format, acceptable by tuple_array_create */
  smt_astt tuple_array_create(const expr2tc &expr, smt_sortt domain);

  /** Initialize tracking data for the address space records. This also sets
   *  up the symbols / addresses of 'NULL', '0', and the invalid pointer */
  void init_addr_space_array(void);
  /** Stores handle for the tuple interface. */
  void set_tuple_iface(tuple_iface *iface);
  /** Stores handle for the array interface. */
  void set_array_iface(array_iface *iface);
  /** Store a new address-allocation record into the address space accounting.
   *  idx indicates the object number of this record. */
  void bump_addrspace_array(unsigned int idx, const expr2tc &val);
  /** Get the symbol name for the current address-allocation record array. */
  std::string get_cur_addrspace_ident(void);
  /** Create and assert address space constraints on the given object ID
   *  number. Essentially, this asserts that all the objects to date don't
   *  overlap with /this/ one. */
  void finalize_pointer_chain(unsigned int obj_num);

  /** Typecast data to bools */
  smt_astt convert_typecast_to_bool(const typecast2t &cast);
  /** Typecast to a fixedbv in bitvector mode */
  smt_astt convert_typecast_to_fixedbv_nonint(const expr2tc &cast);
  smt_astt convert_typecast_to_fixedbv_nonint_from_bv(const expr2tc &cast);
  smt_astt convert_typecast_to_fixedbv_nonint_from_bool(const expr2tc &cast);
  smt_astt convert_typecast_to_fixedbv_nonint_from_fixedbv(const expr2tc &cast);
  /** Typecast anything to an integer (but not pointers) */
  smt_astt convert_typecast_to_ints(const typecast2t &cast);
  smt_astt convert_typecast_to_ints_intmode(const typecast2t &cast);
  smt_astt convert_typecast_to_ints_from_fbv_sint(const typecast2t &cast);
  smt_astt convert_typecast_to_ints_from_unsigned(const typecast2t &cast);
  smt_astt convert_typecast_to_ints_from_bool(const typecast2t &cast);
  /** Typecast something (i.e. an integer) to a pointer */
  smt_astt convert_typecast_to_ptr(const typecast2t &cast);
  /** Typecast a pointer to an integer */
  smt_astt convert_typecast_from_ptr(const typecast2t &cast);
  /** Typecast structs to other structs */
  smt_astt convert_typecast_to_struct(const typecast2t &cast);
  /** Despatch a typecast expression to a more specific typecast method */
  smt_astt convert_typecast(const expr2tc &expr);
  /** Round a real to an integer; not straightforwards at all. */
  smt_astt round_real_to_int(smt_astt a);
  /** Round a fixedbv to an integer. */
  smt_astt round_fixedbv_to_int(smt_astt a, unsigned int width,
                                      unsigned int towidth);

  /** Extract a type definition (i.e. a struct_union_data object) from a type.
   *  This method abstracts the fact that a pointer type is in fact a tuple. */
  const struct_union_data &get_type_def(const type2tc &type) const;
  /** Prep call for creating a tuple array */
  smt_astt tuple_array_create_despatch(const expr2tc &expr, smt_sortt domain);

  /** Convert a boolean to a bitvector with one bit. */
  smt_astt make_bool_bit(smt_astt a);
  /** Convert a bitvector with one bit to boolean. */
  smt_astt make_bit_bool(smt_astt a);

  /** Given an array index, extract the lower n bits of it, where n is the
   *  bitwidth of the array domain. */
  expr2tc fix_array_idx(const expr2tc &idx, const type2tc &array_type);
  /** Convert the size of an array to its bit width. Essential log2 with
   *  some rounding. */
  unsigned long size_to_bit_width(unsigned long sz);
  /** Given an array type, calculate the domain bitwidth it should have. For
   *  nondeterministically or infinite sized arrays, this defaults to the
   *  machine integer width. */
  unsigned long calculate_array_domain_width(const array_type2t &arr);
  /** Given an array type, create an smt sort representing its domain. */
  smt_sortt make_array_domain_sort(const array_type2t &arr);
  /** Like make_array_domain_sort, but a type2tc not an smt_sort */
  type2tc make_array_domain_sort_exp(const array_type2t &arr);
  /** Cast the given expression to the domain width of the array in type */
  expr2tc twiddle_index_width(const expr2tc &expr, const type2tc &type);
  /** For a multi-dimensional array, convert the type into a single dimension
   *  array. This works by concatenating the domain widths together into one
   *  large domain. */
  type2tc flatten_array_type(const type2tc &type);
  /** For a multi-dimensional constant array, flatten the actual definition of
   *  it down to a single dimension */
  expr2tc flatten_array_body(const expr2tc &expr);
  /** Get the base subtype of an array, delving through any intermediate
   *  multidimensional arrays. */
  type2tc get_flattened_array_subtype(const type2tc &type);
  /** Fetch the number of elements in an array (the domain). */
  expr2tc array_domain_to_width(const type2tc &type);
  /** Create an array domain sort */
  smt_sortt mk_int_bv_sort(unsigned int width) {
    if (int_encoding)
      return mk_sort(SMT_SORT_INT);
    else
      return mk_sort(SMT_SORT_BV, width, false);
  }

  /** For the given type, replace all instances of a pointer type with the
   *  struct representation of it. */
  void rewrite_ptrs_to_structs(type2tc &type);

  /** When dealing with multi-dimensional arrays, and selecting one element
   *  out of several dimensions, reduce it to an expression on a single
   *  dimensional array, by concatonating the indexes. Works in conjunction
   *  with flatten_array_type. */
  expr2tc decompose_select_chain(const expr2tc &expr, expr2tc &base);
  /** Like decompose_select_chain, but for multidimensional stores. */
  expr2tc decompose_store_chain(const expr2tc &expr, expr2tc &base);

  /** Prepare an array_of expression by flattening its dimensions, if it
   *  has more than one. */
  smt_astt convert_array_of_prep(const expr2tc &expr);
  /** Create an array of pointers; expects the init_val to be null, because
   *  there's no other way to initialize a pointer array in C, AFAIK. */
  smt_astt pointer_array_of(const expr2tc &init_val,
                                  unsigned long array_width);

  /** Given a textual representation of a real, as one number divided by
   *  another, create a fixedbv representation of it. For use in counterexample
   *  formatting. */
  std::string get_fixed_point(const unsigned width, std::string value) const;

  unsigned int get_member_name_field(const type2tc &t, const irep_idt &name) const;
  unsigned int get_member_name_field(const type2tc &t, const expr2tc &name) const;

  // Ours:
  /** Given an array expression, attempt to extract its valuation from the
   *  solver model, computing a constant_array2tc by calling get_array_elem. */
  expr2tc get_array(smt_astt array, const type2tc &t);

  void delete_all_asts();

  /** @} */

  // Types

  // Type for (optional) AST cache

  struct smt_cache_entryt {
    const expr2tc val;
    smt_astt ast;
    unsigned int level;
  };

  typedef boost::multi_index_container<
    smt_cache_entryt,
    boost::multi_index::indexed_by<
      boost::multi_index::hashed_unique<
        BOOST_MULTI_INDEX_MEMBER(smt_cache_entryt, const expr2tc, val)
      >,
      boost::multi_index::ordered_non_unique<
        BOOST_MULTI_INDEX_MEMBER(smt_cache_entryt, unsigned int, level),
        std::greater<unsigned int>
      >
    >
  > smt_cachet;

  typedef hash_map_cont<type2tc, smt_sortt, type2_hash> smt_sort_cachet;

  // Members
  /** Number of un-popped context pushes encountered so far. */
  unsigned int ctx_level;

  /** A cache mapping expressions to converted SMT ASTs. */
  smt_cachet smt_cache;
  /** A cache of converted type2tc's to smt sorts */
  smt_sort_cachet sort_cache;
  /** Pointer_logict object, which contains some code for formatting how
   *  pointers are displayed in counter-examples. This is a list so that we
   *  can push and pop data when context push/pop operations occur. */
  std::list<pointer_logict> pointer_logic;
  /** Constant struct representing the implementation of the pointer type --
   *  i.e., the struct type that pointers get translated to. */
  type2tc pointer_struct;
  /** Raw pointer to the type2t in pointer_struct, for convenience. */
  const struct_type2t *pointer_type_data; // ptr of pointer_struct
  /** The type of the machine integer type. */
  type2tc machine_int;
  /** The type of the machine unsigned integer type. */
  type2tc machine_uint;
  /** The type of the machine integer that can store a pointer. */
  type2tc machine_ptr;
  /** The SMT sort of this machines integer type. */
  smt_sortt machine_int_sort;
  /** The SMT sort of this machines unsigned integer type. */
  smt_sortt machine_uint_sort;
  /** Sort for booleans. For fast access. */
  smt_sortt boolean_sort;
  /** Whether we are encoding expressions in integer mode or not. */
  bool int_encoding;
  /** A namespace containing all the types in the program. Used to resolve the
   *  rare case where we're doing some pointer arithmetic and need to have the
   *  concrete type of a pointer. */
  const namespacet &ns;

  bool ptr_foo_inited;
  /** Full name of the '__ESBMC_is_dynamic' modelling array. The memory space
   *  stuff may make some assertions using it, see the discussion in the
   *  constructor. */
  std::string dyn_info_arr_name;

  smt_astt null_ptr_ast;
  smt_astt invalid_ptr_ast;

  /** Mapping of name prefixes to use counts: when we want a fresh new name
   *  with a particular prefix, this map stores how many times that prefix has
   *  been used, and thus what number should be appended to make the name
   *  unique. */
  std::map<std::string, unsigned int> fresh_map;

  /** Integer recording how many times the address space allocation record
   *  array has been modified. Essentially, this is like the SSA variable
   *  number, for an array we build / modify at conversion time. In a list so
   *  that we can support push/pop operations. */
  std::list<unsigned int> addr_space_sym_num;
  /** Type of the address space allocation records. Currently a start address
   *  integer and an end address integer. */
  type2tc addr_space_type;
  /** Pointer to type2t object in addr_space_type, for convenience */
  const struct_type2t *addr_space_type_data;
  /** Type of the array of address space allocation records. */
  type2tc addr_space_arr_type;
  /** List of address space allocation sizes. A map from the object number to
   *  the nubmer of bytes allocated. In a list to support pushing and
   *  popping. */
  std::list<std::map<unsigned, unsigned> > addr_space_data;

  // XXX - push-pop will break here.
  typedef std::map<std::string, smt_astt> renumber_mapt;
  std::vector<renumber_mapt> renumber_map;

  /** Lifetime tracking of smt ast's. When a context is pop'd, all the ASTs
   *  created in that context are freed. */
  std::vector<smt_ast *> live_asts;
  /** Accounting of live_asts for push/pop. Records the number of pointers
   *  contained when a push occurred. On pop, the live_asts vector is reset
   *  back to that point. */
  std::vector<unsigned int> live_asts_sizes;

  tuple_iface *tuple_api;
  array_iface *array_api;

  // Workaround for integer shifts. This is an array of the powers of two,
  // up to 2^64.
  smt_astt int_shift_op_array;

  /** Mapping of SMT function IDs to their names. XXX, incorrect size. */
  static const std::string smt_func_name_table[expr2t::end_expr_id];
};

// Define here to enable inlining
extern inline
smt_ast::smt_ast(smt_convt *ctx, smt_sortt s) : sort(s) {
  assert(sort != NULL);
  ctx->live_asts.push_back(this);
}

#endif /* _ESBMC_PROP_SMT_SMT_CONV_H_ */
