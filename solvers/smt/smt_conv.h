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
  SMT_FUNC_BVINT,
  SMT_FUNC_REAL,
  SMT_FUNC_SYMBOL,

  // Nonterminals
};

class smt_sort {
public:
  smt_sort(smt_sort_kind k);
  ~smt_sort(void);
  smt_sort_kind kind;
};

class smt_ast {
  // Question - should this AST node not be the same as any other expression
  // in ESBMC and just use the existing expr representations?
  // Answer - Perhaps, but there's a semantic shift between what ESBMC uses
  // expressions for and what the SMT language actually /has/, i.e. just some
  // function applications and suchlike. Plus an additional layer of type safety
  // can be applied here that can't in the expression stuff.
  //
  // In particular, things where multiple SMT functions map onto one expression
  // operator are troublesome. This means multiple integer operators for
  // different integer modes, signed and unsigned comparisons, the multitude of
  // things that an address-of can turn into, and so forth.
public:
  smt_ast(const smt_sort *s, smt_func_kind k);
  ~smt_ast();

  const smt_sort *sort;
  smt_func_kind kind;

  // Thought of storing ast arguments here; however what shape this takes
  // depends on the backend sovler, so make that someone elses problem.
  // You might ask why store /anything/ non solver specific here; valid
  // question, I figure the sort and kind are seriously important for debugging.
};

class smt_convt: public prop_convt
{
public:
  smt_convt();
  ~smt_convt();

  virtual void push_ctx(void);
  virtual void pop_ctx(void);

  virtual void assert_lit(const literalt &l) = 0;
};

#endif /* _ESBMC_PROP_SMT_SMT_CONV_H_ */
