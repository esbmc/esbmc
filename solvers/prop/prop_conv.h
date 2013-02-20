/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

//
// convert expression to boolean formula
//

#ifndef CPROVER_PROP_CONV_H
#define CPROVER_PROP_CONV_H

#include <string>
#include <list>

#include <irep2.h>
#include <config.h>
#include <hash_cont.h>
#include <message.h>
#include <threeval.h>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>

#include "literal.h"

class prop_convt : public messaget
{
public:
  explicit prop_convt()
  {
    ctx_level = 0;
  }
  virtual ~prop_convt() { }

  typedef enum { P_SATISFIABLE, P_UNSATISFIABLE, P_ERROR, P_SMTLIB } resultt;

  // Methods to push and pop currently converted solver state
  virtual void push_ctx(void);
  virtual void pop_ctx(void);

  // overloading
  virtual void set_to(const expr2tc &expr, bool value) = 0;
  virtual resultt dec_solve() = 0;

  virtual literalt convert(const expr2tc &expr);

  // get a value from counterexample if not valid
  virtual expr2tc get(const expr2tc &expr) = 0;
  
  virtual void clear_cache()
  {
    cache.clear();
  }
  
  // Literal manipulation routines inhereted from propt.
  virtual literalt land(literalt a, literalt b)=0;
  virtual literalt lor(literalt a, literalt b)=0;
  virtual literalt land(const bvt &bv)=0;
  virtual literalt lor(const bvt &bv)=0;
  virtual literalt lnot(literalt a)=0;
  virtual literalt limplies(literalt a, literalt b)=0;
  virtual void set_equal(literalt a, literalt b);

  virtual void l_set_to(literalt a, bool value)
  {
    set_equal(a, const_literal(value));
  }

  // constraints
  virtual void lcnf(const bvt &bv)=0;

  // variables
  virtual literalt new_variable()=0;
  virtual uint64_t get_no_variables() const=0;

  // solving
  virtual const std::string solver_text()=0;

  // satisfying assignment
  virtual tvt l_get(literalt a)=0;

protected:
  virtual literalt convert_expr(const expr2tc &expr) = 0;
  
  unsigned int ctx_level;

  // cache
  struct lit_cachet {
    const expr2tc val;
    literalt l;
    unsigned int level;
  };

  typedef boost::multi_index_container<
    lit_cachet,
    boost::multi_index::indexed_by<
      boost::multi_index::hashed_unique<
        BOOST_MULTI_INDEX_MEMBER(lit_cachet, const expr2tc, val)
      >,
      boost::multi_index::ordered_non_unique<
        BOOST_MULTI_INDEX_MEMBER(lit_cachet, unsigned int, level),
        std::greater<unsigned int>
      >
    >
  > cachet;

  cachet cache;
  
  virtual void ignoring(const expr2tc &expr);

public:
  const cachet &get_cache() const { return cache; }

  virtual void convert_smt_type(const type2t &type, void *arg);
  virtual void convert_smt_type(const bool_type2t &type, void *arg) = 0;
  virtual void convert_smt_type(const unsignedbv_type2t &type, void *arg) = 0;
  virtual void convert_smt_type(const signedbv_type2t &type, void *arg) = 0;
  virtual void convert_smt_type(const array_type2t &type, void *arg) = 0;
  virtual void convert_smt_type(const pointer_type2t &type, void *arg) = 0;
  virtual void convert_smt_type(const struct_type2t &type, void *arg) =0;
  virtual void convert_smt_type(const union_type2t &type, void *arg) =0;
  virtual void convert_smt_type(const fixedbv_type2t &type, void *arg) =0;

  virtual void convert_smt_expr(const expr2t &expr, void *arg);
};

#endif
