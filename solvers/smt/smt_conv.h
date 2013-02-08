#ifndef _ESBMC_PROP_SMT_SMT_CONV_H_
#define _ESBMC_PROP_SMT_SMT_CONV_H_

#include <irep2.h>
#include <solvers/prop/prop_conv.h>

enum smt_func {
  // Terminals
  SMT_FUNC_INT = 0,
  SMT_FUNC_BVINT,
  SMT_FUNC_REAL,
  SMT_FUNC_SYMBOL,

  // Nonterminals
};

class smt_ast;
class smt_sort;

class smt_convt: public prop_convt
{
public:
  smt_convt();
  ~smt_convt();

  virtual void push_ctx(void);
  virtual void pop_ctx(void);

};

#endif /* _ESBMC_PROP_SMT_SMT_CONV_H_ */
