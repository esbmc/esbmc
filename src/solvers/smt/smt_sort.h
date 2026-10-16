#ifndef SOLVERS_SMT_SMT_SORT_H_
#define SOLVERS_SMT_SMT_SORT_H_

#include <cstdio>
#include <cstdlib>
#include <irep2/irep2_type.h>

/** Identifier for SMT sort kinds
 *  Each different kind of sort (i.e., arrays, bv's, bools, etc) gets its own
 *  identifier. To be able to describe multiple kinds at the same time, they
 *  take binary values, so that they can be used as bits in an integer. */
enum smt_sort_kind
{
  SMT_SORT_INT,
  SMT_SORT_REAL,
  SMT_SORT_BV,
  SMT_SORT_FIXEDBV,
  SMT_SORT_ARRAY,
  SMT_SORT_BOOL,
  SMT_SORT_STRUCT,
  SMT_SORT_BVFP,
  SMT_SORT_FPBV,
  SMT_SORT_BVFP_RM,
  SMT_SORT_FPBV_RM,
};

/** A class for storing an SMT sort.
 *  This class abstractly represents an SMT sort: solver converter classes are
 *  expected to extend this and add fields that store their solvers
 *  representation of the sort. Then, this base class is used as a handle
 *  through the rest of the SMT conversion code.
 *
 *  Only a few pieces of sort information are used to make conversion decisions,
 *  and are thus actually stored in the sort object itself.
 *  @see smt_ast
 */

class smt_sort;
typedef const smt_sort *smt_sortt;

class smt_sort
{
public:
  /** Identifies what /kind/ of sort this is.
   *  The specific sort itself may be parameterised with widths and domains,
   *  for example. */
  smt_sort_kind id;

  smt_sort(smt_sort_kind i)
    : id(i), data_width(0), secondary_width(0), range_sort(nullptr)
  {
    assert(id != SMT_SORT_ARRAY);
  }

  smt_sort(smt_sort_kind i, const type2tc &type)
    : id(i),
      data_width(0),
      secondary_width(0),
      range_sort(nullptr),
      tupletype(type)
  {
    assert(i == SMT_SORT_STRUCT);
  }

  smt_sort(smt_sort_kind i, std::size_t width)
    : id(i), data_width(width), secondary_width(0), range_sort(nullptr)
  {
    assert(id >= SMT_SORT_BV || id <= SMT_SORT_FIXEDBV);
  }

  smt_sort(smt_sort_kind i, std::size_t width, std::size_t sigwidth)
    : id(i), data_width(width), secondary_width(sigwidth), range_sort(nullptr)
  {
    assert(id == SMT_SORT_BVFP || id == SMT_SORT_FPBV);
  }

  smt_sort(smt_sort_kind i, std::size_t dom_width, smt_sortt range_sort)
    : id(i), data_width(dom_width), secondary_width(0), range_sort(range_sort)
  {
    assert(id == SMT_SORT_ARRAY);
  }

  smt_sort(
    smt_sort_kind i,
    const type2tc &type,
    std::size_t dom_width,
    smt_sortt range_sort)
    : id(i),
      data_width(dom_width),
      secondary_width(0),
      range_sort(range_sort),
      tupletype(type)
  {
    assert(i == SMT_SORT_ARRAY);
  }

  size_t get_data_width() const
  {
    if (id == SMT_SORT_ARRAY)
      return data_width * range_sort->data_width;
    return data_width;
  }

  size_t get_domain_width() const
  {
    assert(id == SMT_SORT_ARRAY);
    return data_width;
  }

  smt_sortt get_range_sort() const
  {
    assert(id == SMT_SORT_ARRAY);
    assert(range_sort != nullptr);
    return range_sort;
  }

  size_t get_significand_width() const
  {
    assert(id == SMT_SORT_BVFP || id == SMT_SORT_FPBV);
    return secondary_width;
  }

  size_t get_exponent_width() const
  {
    assert(id == SMT_SORT_BVFP || id == SMT_SORT_FPBV);
    std::size_t exp_top = get_data_width() - 2;
    std::size_t exp_bot = get_significand_width() - 2;
    return (exp_top - exp_bot);
  }

  const type2tc &get_tuple_type() const
  {
    assert(!is_nil_type(tupletype));
    return tupletype;
  }

  /** True iff this sort is a `solver_smt_sort<S>` carrying a per-solver
   *  native sort handle in its `s` field — i.e. safe to feed to
   *  `to_solver_smt_sort<S>` and then to a backend's mk_array_sort,
   *  mk_const, mk_array_symbol, etc.  False for the bare `smt_sort`
   *  instances that the tuple flatteners (smt_tuple_node.cpp,
   *  smt_tuple_sym.cpp) construct for SMT_SORT_STRUCT and
   *  SMT_SORT_ARRAY-of-tuple sorts; those should only flow back
   *  through the tuple_api codepath.
   *
   *  Used by to_solver_smt_sort below to convert a silent SIGSEGV
   *  (downcasting a bare smt_sort and reading garbage at the `s`
   *  field offset) into a labeled abort. */
  bool is_solver_native() const
  {
    return solver_native;
  }

  virtual ~smt_sort() = default;

protected:
  bool solver_native = false;

private:
  /** Data size of the sort.
   * For bitvectors and floating-points, this is the bit width,
   * for arrays, the domain (index) bit width,
   * For everything else, undefined */
  size_t data_width;

  /** Secondary width
   * For floating-points, this is the significand width,
   * For everything else, undefined */
  size_t secondary_width;

  /** Range sort
   * For arrays, this is the type of the element
   * For everything else, undefined */
  smt_sortt range_sort;

  /** Type of the tuple
   * For structs, this is the actual type (struct or array of structs) of a
   * tuple that's been flattened For everything else, undefined
   */
  const type2tc tupletype;
};

template <typename solver_sort>
class solver_smt_sort : public smt_sort
{
public:
  solver_smt_sort(smt_sort_kind i, solver_sort _s) : smt_sort(i), s(_s)
  {
    this->solver_native = true;
  }

  solver_smt_sort(smt_sort_kind i, solver_sort _s, const type2tc &_tupletype)
    : smt_sort(i, _tupletype), s(_s)
  {
    this->solver_native = true;
  }

  solver_smt_sort(smt_sort_kind i, solver_sort _s, unsigned int w)
    : smt_sort(i, w), s(_s)
  {
    this->solver_native = true;
  }

  solver_smt_sort(
    smt_sort_kind i,
    solver_sort _s,
    unsigned int w,
    unsigned int sw)
    : smt_sort(i, w, sw), s(_s)
  {
    this->solver_native = true;
  }

  solver_smt_sort(
    smt_sort_kind i,
    solver_sort _s,
    std::size_t dw,
    const smt_sort *_rangesort)
    : smt_sort(i, dw, _rangesort), s(_s)
  {
    this->solver_native = true;
  }

  solver_smt_sort(
    smt_sort_kind i,
    solver_sort _s,
    const type2tc &type,
    std::size_t width,
    smt_sortt range_sort)
    : smt_sort(i, type, width, range_sort), s(_s)
  {
    this->solver_native = true;
  }

  ~solver_smt_sort() override = default;

  solver_sort s;
};

#ifdef NDEBUG
#  define dynamic_cast static_cast
#endif
template <typename T>
const solver_smt_sort<T> *to_solver_smt_sort(smt_sortt s)
{
  /* Phase-0 diagnostic floor: a bare smt_sort (created by the tuple
   * flatteners in src/solvers/smt/tuple/) carries no `s` field.  In a
   * NDEBUG build the dynamic_cast macro reduces to static_cast and the
   * unsafe downcast silently reads garbage where `s` would be in the
   * derived layout — bitwuzla/cvc5/etc. then dereference the garbage
   * and SIGSEGV.  Check the discriminator FIRST so the failure mode is
   * a labeled abort instead of a stack-walk-from-null.  */
  if (!s->is_solver_native())
  {
    std::fprintf(
      stderr,
      "ESBMC internal error: bare smt_sort (id=%d) reached "
      "to_solver_smt_sort<>; this sort was produced by the tuple "
      "flattener and must only flow back through the tuple_api "
      "codepath, not into a backend's mk_array_sort/mk_const/"
      "mk_array_symbol.  Either the carve-out at smt_conv.cpp's "
      "array_id case is missing a shape, or a new bare-sort producer "
      "was added without a matching consumer-side route.\n",
      static_cast<int>(s->id));
    std::abort();
  }
  const solver_smt_sort<T> *r = dynamic_cast<const solver_smt_sort<T> *>(s);
  assert(r);
  return r;
}
#ifdef dynamic_cast
#  undef dynamic_cast
#endif

#endif /* SOLVERS_SMT_SMT_SORT_H_ */
