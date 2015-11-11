/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_STD_CODE_H
#define CPROVER_STD_CODE_H

#include <assert.h>

#include <expr.h>

class codet:public exprt
{
public:
  codet():exprt("code", typet("code"))
  {
  }
  
  codet(const irep_idt &statement):exprt("code", typet("code"))
  {
    set_statement(statement);
  }
  
  void set_statement(const irep_idt &statement)
  {
    this->statement(statement);
  }

  const irep_idt &get_statement() const
  {
    return get("statement");
  }
  
  codet &first_statement();
  const codet &first_statement() const;
  codet &last_statement();
  const codet &last_statement() const;
  class code_blockt &make_block();
};

extern inline const codet &to_code(const exprt &expr)
{
  assert(expr.is_code());
  return static_cast<const codet &>(expr);
}

extern inline codet &to_code(exprt &expr)
{
  assert(expr.is_code());
  return static_cast<codet &>(expr);
}

class code_blockt:public codet
{
public:
  code_blockt():codet("block")
  {
  }
};

extern inline const code_blockt &to_code_block(const codet &code)
{
  assert(code.get_statement()=="block");
  return static_cast<const code_blockt &>(code);
}

extern inline code_blockt &to_code_block(codet &code)
{
  assert(code.get_statement()=="block");
  return static_cast<code_blockt &>(code);
}

class code_skipt:public codet
{
public:
  code_skipt():codet("skip")
  {
  }
};

class code_assignt:public codet
{
public:
  code_assignt():codet("assign")
  {
    operands().resize(2);
  }
  
  code_assignt(const exprt &lhs, const exprt &rhs):codet("assign")
  {
    copy_to_operands(lhs, rhs);
  }
  
  exprt &lhs()
  {
    return op0();
  }

  exprt &rhs()
  {
    return op1();
  }

  const exprt &lhs() const
  {
    return op0();
  }

  const exprt &rhs() const
  {
    return op1();
  }
};

extern inline const code_assignt &to_code_assign(const codet &code)
{
  assert(code.get_statement()=="assign");
  return static_cast<const code_assignt &>(code);
}

extern inline code_assignt &to_code_assign(codet &code)
{
  assert(code.get_statement()=="assign");
  return static_cast<code_assignt &>(code);
}

class code_declt:public codet
{
public:
  code_declt():codet("decl")
  {
    operands().reserve(2);
  }
  
  explicit code_declt(const exprt &lhs):codet("decl")
  {
    copy_to_operands(lhs);
  }
  
  code_declt(const exprt &lhs, const exprt &rhs):codet("decl")
  {
    copy_to_operands(lhs, rhs);
  }
  
  exprt &lhs()
  {
    return op0();
  }

  exprt &rhs()
  {
    return op1();
  }

  const exprt &lhs() const
  {
    return op0();
  }

  const exprt &rhs() const
  {
    return op1();
  }

  friend inline const code_declt &to_code_decl(const codet &code)
  {
    assert(code.get_statement()=="decl");
    return static_cast<const code_declt &>(code);
  }

  friend inline code_declt &to_code_decl(codet &code)
  {
    assert(code.get_statement()=="decl");
    return static_cast<code_declt &>(code);
  }

};

const code_declt &to_code_decl(const codet &code);
code_declt &to_code_decl(codet &code);

class code_assumet:public codet
{
public:
  code_assumet():codet("assume")
  {
    operands().reserve(1);
  }
};

extern inline const code_assumet &to_code_assume(const codet &code)
{
  assert(code.get_statement()=="assume");
  return static_cast<const code_assumet &>(code);
}

extern inline code_assumet &to_code_assume(codet &code)
{
  assert(code.get_statement()=="assume");
  return static_cast<code_assumet &>(code);
}

class code_assertt:public codet
{
public:
  code_assertt():codet("assert")
  {
    operands().reserve(1);
  }
};

extern inline const code_assertt &to_code_assert(const codet &code)
{
  assert(code.get_statement()=="assert");
  return static_cast<const code_assertt &>(code);
}

extern inline code_assertt &to_code_assert(codet &code)
{
  assert(code.get_statement()=="assert");
  return static_cast<code_assertt &>(code);
}

class code_ifthenelset:public codet
{
public:
  code_ifthenelset():codet("ifthenelse")
  {
    operands().reserve(3);
  }
  
  const exprt &cond() const
  {
    return op0();
  }
};

extern inline const code_ifthenelset &to_code_ifthenelse(const codet &code)
{
  assert(code.get_statement()=="ifthenelse");
  return static_cast<const code_ifthenelset &>(code);
}

extern inline code_ifthenelset &to_code_ifthenelse(codet &code)
{
  assert(code.get_statement()=="ifthenelse");
  return static_cast<code_ifthenelset &>(code);
}

class code_function_callt:public codet
{
public:
  code_function_callt():codet("function_call")
  {
    operands().resize(3);
    lhs().make_nil();
    op2().id("arguments");
  }
  
  exprt &lhs()
  {
    return op0();
  }

  const exprt &lhs() const
  {
    return op0();
  }

  exprt &function()
  {
    return op1();
  }

  const exprt &function() const
  {
    return op1();
  }

  typedef exprt::operandst argumentst;

  argumentst &arguments()
  {
    return op2().operands();
  }

  const argumentst &arguments() const
  {
    return op2().operands();
  }
};

extern inline const code_function_callt &to_code_function_call(const codet &code)
{
  assert(code.get_statement()=="function_call");
  return static_cast<const code_function_callt &>(code);
}

extern inline code_function_callt &to_code_function_call(codet &code)
{
  assert(code.get_statement()=="function_call");
  return static_cast<code_function_callt &>(code);
}

class code_returnt:public codet
{
public:
  code_returnt():codet("return")
  {
    operands().reserve(1);
  }
  
  const exprt &return_value() const
  {
    return op0();
  }
  
  exprt &return_value()
  {
    operands().resize(1);
    return op0();
  }
  
  bool has_return_value() const
  {
    return operands().size()==1;
  }
};

extern inline const code_returnt &to_code_return(const codet &code)
{
  assert(code.get_statement()=="return");
  return static_cast<const code_returnt &>(code);
}

extern inline code_returnt &to_code_return(codet &code)
{
  assert(code.get_statement()=="return");
  return static_cast<code_returnt &>(code);
}

class code_labelt:public codet
{
public:
  code_labelt():codet("label")
  {
    operands().resize(1);
  }

  inline bool is_default() const
  {
    return dfault();
  }

  const exprt::operandst &case_op() const
  {
    return static_cast<const exprt &>(case_irep()).operands();
  }
  
  const irep_idt &get_label() const
  {
    return get("label");
  }

  void set_label(const irep_idt &label)
  {
    this->label(label);
  }
};

extern inline const code_labelt &to_code_label(const codet &code)
{
  assert(code.get_statement()=="label");
  return static_cast<const code_labelt &>(code);
}

extern inline code_labelt &to_code_label(codet &code)
{
  assert(code.get_statement()=="label");
  return static_cast<code_labelt &>(code);
}

class code_breakt:public codet
{
public:
  code_breakt():codet("break")
  {
  }
};

extern inline const code_breakt &to_code_break(const codet &code)
{
  assert(code.get_statement()=="break");
  return static_cast<const code_breakt &>(code);
}

extern inline code_breakt &to_code_break(codet &code)
{
  assert(code.get_statement()=="break");
  return static_cast<code_breakt &>(code);
}

class code_continuet:public codet
{
public:
  code_continuet():codet("break")
  {
  }
};

extern inline const code_continuet &to_code_continue(const codet &code)
{
  assert(code.get_statement()=="continue");
  return static_cast<const code_continuet &>(code);
}

extern inline code_continuet &to_code_continue(codet &code)
{
  assert(code.get_statement()=="continue");
  return static_cast<code_continuet &>(code);
}

class code_expressiont:public codet
{
public:
  code_expressiont():codet("expression")
  {
    operands().reserve(1);
  }
  
  friend code_expressiont &to_code_expression(codet &code)
  {
    assert(code.get_statement()=="expression");
    return static_cast<code_expressiont &>(code);
  }

  friend const code_expressiont &to_code_expression(const codet &code)
  {
    assert(code.get_statement()=="expression");
    return static_cast<const code_expressiont &>(code);
  }

  inline const exprt &expression() const
  {
    return op0();
  }

  inline exprt &expression()
  {
    return op0();
  }
};

code_expressiont &to_code_expression(codet &code);
const code_expressiont &to_code_expression(const codet &code);

class side_effect_exprt:public exprt
{
public:
  explicit side_effect_exprt(const irep_idt &statement):exprt("sideeffect")
  {
    set_statement(statement);
  }

  friend side_effect_exprt &to_side_effect_expr(exprt &expr)
  {
    assert(expr.id()=="sideeffect");
    return static_cast<side_effect_exprt &>(expr);
  }

  friend const side_effect_exprt &to_side_effect_expr(const exprt &expr)
  {
    assert(expr.id()=="sideeffect");
    return static_cast<const side_effect_exprt &>(expr);
  }
  
  const irep_idt &get_statement() const
  {
    return get("statement");
  }

  void set_statement(const irep_idt &statement)
  {
    return this->statement(statement);
  }
};

side_effect_exprt &to_side_effect_expr(exprt &expr);
const side_effect_exprt &to_side_effect_expr(const exprt &expr);

class side_effect_expr_nondett:public side_effect_exprt
{
public:
  side_effect_expr_nondett():side_effect_exprt("nondet")
  {
  }

  explicit side_effect_expr_nondett(const typet &t):side_effect_exprt("nondet")
  {
    type()=t;
  }
};

class side_effect_expr_function_callt:public side_effect_exprt
{
public:
  side_effect_expr_function_callt():side_effect_exprt("function_call")
  {
    operands().resize(2);
    op1().id("arguments");
  }

  exprt &function()
  {
    return op0();
  }

  const exprt &function() const
  {
    return op0();
  }

  exprt::operandst &arguments()
  {
    return op1().operands();
  }

  const exprt::operandst &arguments() const
  {
    return op1().operands();
  }

  friend side_effect_expr_function_callt &to_side_effect_expr_function_call(exprt &expr)
  {
    assert(expr.id()=="sideeffect");
    assert(expr.statement()=="function_call");
    return static_cast<side_effect_expr_function_callt &>(expr);
  }

  friend const side_effect_expr_function_callt &to_side_effect_expr_function_call(const exprt &expr)
  {
    assert(expr.id()=="sideeffect");
    assert(expr.statement()=="function_call");
    return static_cast<const side_effect_expr_function_callt &>(expr);
  }
};

side_effect_expr_function_callt &to_side_effect_expr_function_call(exprt &expr);
const side_effect_expr_function_callt &to_side_effect_expr_function_call(const exprt &expr);

#endif
