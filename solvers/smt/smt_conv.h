#ifndef _ESBMC_PROP_SMT_SMT_CONV_H_
#define _ESBMC_PROP_SMT_SMT_CONV_H_

#include <irep2.h>
#include <solvers/prop/prop_conv.h>

enum smt_sort_kind {
  SMT_SORT_INT,
  SMT_SORT_REAL,
  SMT_SORT_BV,
  SMT_SORT_ARRAY,
  SMT_SORT_BOOL,
};

enum smt_func_kind {
  // Terminals
  SMT_FUNC_INT = 0,
  SMT_FUNC_HACKS, // indicate the solver /has/ to use the temp expr.
  SMT_FUNC_BVINT,
  SMT_FUNC_REAL,
  SMT_FUNC_SYMBOL,

  // Nonterminals
  SMT_FUNC_ADD,
  SMT_FUNC_BVADD,
  SMT_FUNC_SUB,
  SMT_FUNC_BVSUB,
  SMT_FUNC_MUL,
  SMT_FUNC_DIV,
  SMT_FUNC_BVDIV,
};

class smt_ast {
  // Completely opaque class for storing whatever data the backend solver feels
  // is appropriate. I orignally thought this should contain useful tracking
  // data for what kind of AST this is, but then realised that this is just
  // overhead and whatever the solver stores should be enough for that too
  //
  // Question - should this AST node (or what it represents) not be the same as
  // any other expression in ESBMC and just use the existing expr
  // representations?  Answer - Perhaps, but there's a semantic shift between
  // what ESBMC uses expressions for and what the SMT language actually /has/,
  // i.e. just some function applications and suchlike. Plus an additional
  // layer of type safety can be applied here that can't in the expression
  // stuff.
  //
  // In particular, things where multiple SMT functions map onto one expression
  // operator are troublesome. This means multiple integer operators for
  // different integer modes, signed and unsigned comparisons, the multitude of
  // things that an address-of can turn into, and so forth.
};

class smt_sort {
  // Same story as smt_ast.
};

class smt_convt: public prop_convt
{
public:
  smt_convt(bool enable_cache);
  ~smt_convt();

  virtual void push_ctx(void);
  virtual void pop_ctx(void);

  virtual void assert_lit(const literalt &l) = 0;

  virtual smt_ast *mk_func_app(const smt_sort *s, smt_func_kind k,
                               smt_ast **args, unsigned int numargs,
                               const expr2tc &temp) = 0;
  virtual smt_sort *mk_sort(const smt_sort_kind k, ...) = 0;
  virtual literalt mk_lit(const smt_ast *s) = 0;
  virtual smt_ast *mk_smt_int(const mp_integer &theint) = 0;
  virtual smt_ast *mk_smt_real(const mp_integer &thereal) = 0;
  virtual smt_ast *mk_smt_bvint(const mp_integer &theint, unsigned int w) = 0;
  virtual smt_ast *mk_smt_bool(bool val) = 0;
  virtual smt_ast *mk_smt_symbol(const std::string &name, const smt_sort *s) =0;

  virtual void set_to(const expr2tc &expr, bool value);
  virtual literalt convert_expr(const expr2tc &expr);

  // Types

  // Types for union map.
  struct union_var_mapt {
    std::string ident;
    unsigned int idx;
    unsigned int level;
  };

  typedef boost::multi_index_container<
    union_var_mapt,
    boost::multi_index::indexed_by<
      boost::multi_index::hashed_unique<
        BOOST_MULTI_INDEX_MEMBER(union_var_mapt, std::string, ident)
      >,
      boost::multi_index::ordered_non_unique<
        BOOST_MULTI_INDEX_MEMBER(union_var_mapt, unsigned int, level),
        std::greater<unsigned int>
      >
    >
  > union_varst;

  // Type for (optional) AST cache

  struct smt_cache_entryt {
    const expr2tc val;
    const smt_ast *ast;
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

  // Members
  union_varst union_vars;
  smt_cachet smt_cache;
  bool caching;
};

#endif /* _ESBMC_PROP_SMT_SMT_CONV_H_ */
