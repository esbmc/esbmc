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

  inline explicit code_assumet(const exprt &expr):codet("assume")
  {
    copy_to_operands(expr);
  }

  inline const exprt &assumption() const
  {
    return op0();
  }

  inline exprt &assumption()
  {
    return op0();
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

  inline explicit code_assertt(const exprt &expr):codet("assert")
  {
    copy_to_operands(expr);
  }

  inline const exprt &assertion() const
  {
    return op0();
  }

  inline exprt &assertion()
  {
    return op0();
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

/*! \brief A `switch' instruction
*/
class code_switcht:public codet
{
public:
  inline code_switcht():codet("switch")
  {
    operands().resize(2);
  }

  inline const exprt &value() const
  {
    return op0();
  }

  inline exprt &value()
  {
    return op0();
  }

  inline const codet &body() const
  {
    return to_code(op1());
  }

  inline codet &body()
  {
    return static_cast<codet &>(op1());
  }
};

static inline const code_switcht &to_code_switch(const codet &code)
{
  assert(code.get_statement()=="switch" &&
         code.operands().size()==2);
  return static_cast<const code_switcht &>(code);
}

static inline code_switcht &to_code_switch(codet &code)
{
  assert(code.get_statement()=="switch" &&
         code.operands().size()==2);
  return static_cast<code_switcht &>(code);
}

/*! \brief A `while' instruction
*/
class code_whilet:public codet
{
public:
  inline code_whilet():codet("while")
  {
    operands().resize(2);
  }

  inline const exprt &cond() const
  {
    return op0();
  }

  inline exprt &cond()
  {
    return op0();
  }

  inline const codet &body() const
  {
    return to_code(op1());
  }

  inline codet &body()
  {
    return static_cast<codet &>(op1());
  }
};

static inline const code_whilet &to_code_while(const codet &code)
{
  assert(code.get_statement()=="while" &&
         code.operands().size()==2);
  return static_cast<const code_whilet &>(code);
}

static inline code_whilet &to_code_while(codet &code)
{
  assert(code.get_statement()=="while" &&
         code.operands().size()==2);
  return static_cast<code_whilet &>(code);
}

/*! \brief A `do while' instruction
*/
class code_dowhilet:public codet
{
public:
  inline code_dowhilet():codet("dowhile")
  {
    operands().resize(2);
  }

  inline const exprt &cond() const
  {
    return op0();
  }

  inline exprt &cond()
  {
    return op0();
  }

  inline const codet &body() const
  {
    return to_code(op1());
  }

  inline codet &body()
  {
    return static_cast<codet &>(op1());
  }
};

static inline const code_dowhilet &to_code_dowhile(const codet &code)
{
  assert(code.get_statement()=="dowhile" &&
         code.operands().size()==2);
  return static_cast<const code_dowhilet &>(code);
}

static inline code_dowhilet &to_code_dowhile(codet &code)
{
  assert(code.get_statement()=="dowhile" &&
         code.operands().size()==2);
  return static_cast<code_dowhilet &>(code);
}

/*! \brief A `for' instruction
*/
class code_fort:public codet
{
public:
  inline code_fort():codet("for")
  {
    operands().resize(4);
  }

  // nil or a statement
  inline const exprt &init() const
  {
    return op0();
  }

  inline exprt &init()
  {
    return op0();
  }

  inline const exprt &cond() const
  {
    return op1();
  }

  inline exprt &cond()
  {
    return op1();
  }

  inline const exprt &iter() const
  {
    return op2();
  }

  inline exprt &iter()
  {
    return op2();
  }

  inline const codet &body() const
  {
    return to_code(op3());
  }

  inline codet &body()
  {
    return static_cast<codet &>(op3());
  }
};

static inline const code_fort &to_code_for(const codet &code)
{
  assert(code.get_statement()=="for" &&
         code.operands().size()==4);
  return static_cast<const code_fort &>(code);
}

static inline code_fort &to_code_for(codet &code)
{
  assert(code.get_statement()=="for" &&
         code.operands().size()==4);
  return static_cast<code_fort &>(code);
}

/*! \brief A `goto' instruction
*/
class code_gotot:public codet
{
public:
  inline code_gotot():codet("goto")
  {
  }

  explicit inline code_gotot(const irep_idt &label):codet("goto")
  {
    set_destination(label);
  }

  void set_destination(const irep_idt &label)
  {
    set("destination", label);
  }

  const irep_idt &get_destination() const
  {
    return get("destination");
  }
};

static inline const code_gotot &to_code_goto(const codet &code)
{
  assert(code.get_statement()=="goto" &&
         code.operands().empty());
  return static_cast<const code_gotot &>(code);
}

static inline code_gotot &to_code_goto(codet &code)
{
  assert(code.get_statement()=="goto" &&
         code.operands().empty());
  return static_cast<code_gotot &>(code);
}

/*! \brief A function call

    The function call instruction has three operands.
    The first is the expression that is used to store
    the return value. The second is the function called.
    The third is a vector of argument values.
*/
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

  codet &code()
  {
    return static_cast<codet &>(op0());
  }

  const codet &code() const
  {
    return static_cast<const codet &>(op0());
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

/*! \brief A switch-case
*/
class code_switch_caset:public codet
{
public:
  inline code_switch_caset():codet("switch_case")
  {
    operands().resize(2);
  }

  inline code_switch_caset(
    const exprt &_case_op, const codet &_code) : codet("switch_case")
  {
    copy_to_operands(_case_op, _code);
  }

  inline bool is_default() const
  {
    return dfault();
  }

  inline void set_default(bool value)
  {
    return dfault(value);
  }

  inline const exprt &case_op() const
  {
    return op0();
  }

  inline exprt &case_op()
  {
    return op0();
  }

  codet &code()
  {
    return static_cast<codet &>(op1());
  }

  const codet &code() const
  {
    return static_cast<const codet &>(op1());
  }
};

static inline const code_switch_caset &to_code_switch_case(const codet &code)
{
  assert(code.get_statement()=="switch_case" && code.operands().size()==2);
  return static_cast<const code_switch_caset &>(code);
}

static inline code_switch_caset &to_code_switch_case(codet &code)
{
  assert(code.get_statement()=="switch_case" && code.operands().size()==2);
  return static_cast<code_switch_caset &>(code);
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
  code_continuet():codet("continue")
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

  inline side_effect_exprt(const irep_idt &statement, const typet &_type)
    : exprt("sideeffect", _type)
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

  side_effect_expr_function_callt(const typet &_type)
    : side_effect_exprt("function_call", _type)
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
