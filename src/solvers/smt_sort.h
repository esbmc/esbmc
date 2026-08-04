#ifndef SOLVERS_SMT_SMT_SORT_H_
#define SOLVERS_SMT_SMT_SORT_H_

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

  smt_sort(smt_sort_kind i) : id(i), data_width(0), range_sort(nullptr)
  {
    assert(id != SMT_SORT_ARRAY);
  }

  smt_sort(smt_sort_kind i, std::size_t width)
    : id(i), data_width(width), range_sort(nullptr)
  {
    assert(
      id == SMT_SORT_BV || id == SMT_SORT_BOOL || id == SMT_SORT_FPBV_RM ||
      id == SMT_SORT_BVFP_RM);
  }

  smt_sort(smt_sort_kind i, std::size_t dom_width, smt_sortt range_sort)
    : id(i), data_width(dom_width), range_sort(range_sort)
  {
    assert(id == SMT_SORT_ARRAY);
  }

  /** True for the Int and Real sorts, i.e. the integer/real encoding's
   *  arithmetic sorts as opposed to bit-vectors. Mirrors camada's
   *  SMTSort::isArithSort(). */
  bool is_arith() const
  {
    return id == SMT_SORT_INT || id == SMT_SORT_REAL;
  }

  /* The remaining kind predicates, named after camada's. */
  bool is_bool() const
  {
    return id == SMT_SORT_BOOL;
  }
  bool is_int() const
  {
    return id == SMT_SORT_INT;
  }
  bool is_bv() const
  {
    return id == SMT_SORT_BV;
  }
  bool is_array() const
  {
    return id == SMT_SORT_ARRAY;
  }
  bool is_tuple() const
  {
    return id == SMT_SORT_STRUCT;
  }
  bool is_fp() const
  {
    return id == SMT_SORT_FPBV;
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

  virtual ~smt_sort() = default;

private:
  /** Data size of the sort.
   * For bitvectors and floating-points, this is the bit width,
   * for arrays, the domain (index) bit width,
   * For everything else, undefined */
  size_t data_width;

  /** Range sort
   * For arrays, this is the type of the element
   * For everything else, undefined */
  smt_sortt range_sort;

};

template <typename solver_sort>
class solver_smt_sort : public smt_sort
{
public:
  solver_smt_sort(smt_sort_kind i, solver_sort _s) : smt_sort(i), s(_s)
  {
  }

  solver_smt_sort(smt_sort_kind i, solver_sort _s, unsigned int w)
    : smt_sort(i, w), s(_s)
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

  ~solver_smt_sort() override = default;

  solver_sort s;
};

#ifdef NDEBUG
#  define dynamic_cast static_cast
#endif
template <typename T>
const solver_smt_sort<T> *to_solver_smt_sort(smt_sortt s)
{
  const solver_smt_sort<T> *r = dynamic_cast<const solver_smt_sort<T> *>(s);
  assert(r);
  return r;
}
#ifdef dynamic_cast
#  undef dynamic_cast
#endif

#endif /* SOLVERS_SMT_SMT_SORT_H_ */
