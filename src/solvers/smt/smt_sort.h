#ifndef SOLVERS_SMT_SMT_SORT_H_
#define SOLVERS_SMT_SMT_SORT_H_

#include <cstdlib>

/** Identifier for SMT sort kinds
 *  Each different kind of sort (i.e. arrays, bv's, bools, etc) gets its own
 *  identifier. To be able to describe multiple kinds at the same time, they
 *  take binary values, so that they can be used as bits in an integer. */
enum smt_sort_kind
{
  SMT_SORT_INT = 1,
  SMT_SORT_REAL = 2,
  SMT_SORT_SBV = 4,
  SMT_SORT_UBV = 8,
  SMT_SORT_FIXEDBV = 16,
  SMT_SORT_ARRAY = 32,
  SMT_SORT_BOOL = 64,
  SMT_SORT_STRUCT = 128,
  SMT_SORT_UNION = 256, // Contencious
  SMT_SORT_FLOATBV = 512,
  SMT_SORT_FLOATBV_RM = 1024
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

  smt_sort(smt_sort_kind i) : id(i), data_width(0), secondary_width(0)
  {
    assert(id != SMT_SORT_ARRAY);
  }

  smt_sort(smt_sort_kind i, std::size_t width)
    : id(i), data_width(width), secondary_width(0)
  {
    assert(width != 0 || i == SMT_SORT_INT);
    assert(id != SMT_SORT_ARRAY);
  }

  smt_sort(smt_sort_kind i, std::size_t rwidth, std::size_t domwidth)
    : id(i), data_width(rwidth), secondary_width(domwidth)
  {
    assert(id == SMT_SORT_ARRAY || id == SMT_SORT_FLOATBV);
    // assert(secondary_width != 0);
    // XXX not applicable during int mode?
  }

  size_t get_data_width() const
  {
    return data_width;
  }

  size_t get_domain_width() const
  {
    assert(id == SMT_SORT_ARRAY);
    return secondary_width;
  }

  size_t get_significand_width() const
  {
    assert(id == SMT_SORT_FLOATBV);
    return secondary_width;
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
   * for arrays this is the width of array domain,
   * For everything else, undefined */
  size_t secondary_width;
};

#endif /* SOLVERS_SMT_SMT_SORT_H_ */
