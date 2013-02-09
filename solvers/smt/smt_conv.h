#ifndef _ESBMC_PROP_SMT_SMT_CONV_H_
#define _ESBMC_PROP_SMT_SMT_CONV_H_

#include <irep2.h>
#include <solvers/prop/prop_conv.h>

enum smt_func_kind {
  // Terminals
  SMT_FUNC_INT = 0,
  SMT_FUNC_BVINT,
  SMT_FUNC_REAL,
  SMT_FUNC_SYMBOL,

  // Nonterminals
};

class smt_sort {
  // Can't currently think what this should contain.
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
  smt_ast(smt_func_kind k, const smt_ast *a);
  smt_ast(smt_func_kind k, const smt_ast *a, const smt_ast *b);
  smt_ast(smt_func_kind k, const smt_ast *a, const smt_ast *b,const smt_ast *c);
  ~smt_ast();

  smt_sort sort;
  smt_func_kind kind;
  const smt_ast *arguments[3]; // No point in making this dynamically sized IMO.
};

class smt_convt: public prop_convt
{
public:
  smt_convt();
  ~smt_convt();

  virtual void push_ctx(void);
  virtual void pop_ctx(void);

};

#endif /* _ESBMC_PROP_SMT_SMT_CONV_H_ */
