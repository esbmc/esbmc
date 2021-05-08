#ifndef SOLVERS_SMT_SMT_SORT_H_
#define SOLVERS_SMT_SMT_SORT_H_

#include <util/irep2_type.h>

/** Identifier for SMT sort kinds
 *  Each different kind of sort (i.e. arrays, bv's, bools, etc) gets its own
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
 *  Only a few piece of sort information are used to make conversion decisions,
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
    if(id == SMT_SORT_ARRAY)
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

  virtual ~smt_sort() = default;

private:
  /** Data size of the sort.
   * For bitvectors and floating-points this is the bit width,
   * for arrays the range BV bit width,
   * For everything else, undefined */
  size_t data_width;

  /** Secondary width
   * For floating-points this is the significand width,
   * For everything else, undefined */
  size_t secondary_width;

  /** Range sort
   * For arrays this is the type of the element
   * For everything else, undefined */
  smt_sortt range_sort;

  /** Type of the tuple
   * For structs this is actual type (struct or array of structs) of a tuple
   * that's been flattened
   * For everything else, undefined
   */
  const type2tc tupletype;
};

template <typename solver_sort>
class solver_smt_sort : public smt_sort
{
public:
  solver_smt_sort(smt_sort_kind i, solver_sort _s) : smt_sort(i), s(_s)
  {
  }

  solver_smt_sort(smt_sort_kind i, solver_sort _s, const type2tc &_tupletype)
    : smt_sort(i, _tupletype), s(_s)
  {
  }

  solver_smt_sort(smt_sort_kind i, solver_sort _s, unsigned int w)
    : smt_sort(i, w), s(_s)
  {
  }

  solver_smt_sort(
    smt_sort_kind i,
    solver_sort _s,
    unsigned int w,
    unsigned int sw)
    : smt_sort(i, w, sw), s(_s)
  {
  }

  solver_smt_sort(
    smt_sort_kind i,
    solver_sort _s,
    std::size_t dw,
    const smt_sort *_rangesort)
    : smt_sort(i, dw, _rangesort), s(_s)
  {
  }

  solver_smt_sort(
    smt_sort_kind i,
    solver_sort _s,
    const type2tc &type,
    std::size_t width,
    smt_sortt range_sort)
    : smt_sort(i, type, width, range_sort), s(_s)
  {
  }

  ~solver_smt_sort() override = default;

  solver_sort s;
};

#ifdef NDEBUG
#define dynamic_cast static_cast
#endif
template <typename T>
const solver_smt_sort<T> *to_solver_smt_sort(smt_sortt s)
{
  return dynamic_cast<const solver_smt_sort<T> *>(s);
}
#ifdef dynamic_cast
#undef dynamic_cast
#endif

#endif /* SOLVERS_SMT_SMT_SORT_H_ */
