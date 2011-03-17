/*******************************************************************\

Module: Conversion of ANSI-C Statements

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <expr_util.h>
#include <prefix.h>
#include <std_code.h>

#include "c_typecast.h"
#include "convert-c.h"

/*******************************************************************\

Function: convert_ct::convert_ExpressionStemnt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_ExpressionStemnt(
  const ExpressionStemnt &statement,
  exprt &dest)
{
  dest.set("statement", "expression");
  dest.operands().resize(1);

  convert(*statement.expression, dest.op0());
}

/*******************************************************************\

Function: convert_ct::convert_ReturnStemnt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_ReturnStemnt(
  const ReturnStemnt &statement,
  exprt &dest)
{
  dest.set("statement", "return");

  if(statement.result!=NULL)
  {
    exprt result;
    convert(*statement.result, result);
    dest.move_to_operands(result);
  }
}

/*******************************************************************\

Function: convert_ct::convert_Decl

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_Decl(
  const Decl &decl,
  const Location &location,
  exprt &dest,
  bool function_parameter)
{
  symbolt symbol;
  
  convert(decl, location, function_parameter, false, symbol);

  if(symbol.name=="")
    return;
    
  codet decl_expr("decl");
  set_location(decl_expr, location);

  decl_expr.copy_to_operands(symbol_expr(symbol));

  if(decl.initializer!=NULL)
  {
    if(symbol.is_static)
      convert(*decl.initializer, symbol.value);
    else
    {
      exprt tmp;
      convert(*decl.initializer, tmp);
      decl_expr.move_to_operands(tmp);
    }
  }

  dest.move_to_operands(decl_expr);
  
  symbols.push_back(symbolt());
  symbols.back().swap(symbol);
}

/*******************************************************************\

Function: convert_ct::convert_DeclStemnt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_DeclStemnt(
  const DeclStemnt &statement,
  exprt &dest)
{
  dest.set("statement", "decl-block");

  dest.reserve_operands(statement.decls.size());
  
  for(DeclVector::const_iterator it=statement.decls.begin();
      it!=statement.decls.end(); it++)
  {
    assert(*it!=NULL);
    convert_Decl(**it, statement.location, dest, false);
  }
}

/*******************************************************************\

Function: convert_ct::convert_ForStemnt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_ForStemnt(
  const ForStemnt &statement,
  exprt &dest)
{
  dest.set("statement", "for");

  dest.reserve_operands(4);

  dest.operands().push_back((exprt &)nil_rep);

  if(statement.init!=NULL)
  {
    exprt tmp("code");
    tmp.set("statement", "expression");
    tmp.operands().resize(1);
  
    convert(*statement.init, tmp.op0());
    dest.operands().back().swap(tmp);
  }

  dest.operands().push_back((exprt &)nil_rep);

  if(statement.cond!=NULL)
    convert(*statement.cond, dest.operands().back());

  dest.operands().push_back((exprt &)nil_rep);

  if(statement.incr!=NULL)
  {
    exprt tmp("code");
    tmp.set("statement", "expression");
    tmp.operands().resize(1);
  
    convert(*statement.incr, tmp.op0());
    dest.operands().back().swap(tmp);
  }

  dest.operands().push_back((exprt &)nil_rep);

  if(statement.block!=NULL)
    convert_Statement(*statement.block, dest.operands().back());
}

/*******************************************************************\

Function: convert_ct::convert_WhileStment

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_WhileStemnt(
  const WhileStemnt &statement,
  exprt &dest)
{
  dest.set("statement", "while");

  dest.reserve_operands(2);

  dest.operands().push_back((exprt &)nil_rep);

  if(statement.cond!=NULL)
    convert(*statement.cond, dest.operands().back());

  dest.operands().push_back((exprt &)nil_rep);

  if(statement.block!=NULL)
    convert_Statement(*statement.block, dest.operands().back());
}

/*******************************************************************\

Function: convert_ct::convert_DoWhileStemnt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_DoWhileStemnt(
  const DoWhileStemnt &statement,
  exprt &dest)
{
  dest.set("statement", "dowhile");

  dest.reserve_operands(2);

  dest.operands().push_back((exprt &)nil_rep);

  if(statement.cond!=NULL)
    convert(*statement.cond, dest.operands().back());

  dest.operands().push_back((exprt &)nil_rep);
  if(statement.block!=NULL)
    convert_Statement(*statement.block, dest.operands().back());
}

/*******************************************************************\

Function: convert_ct::convert_ContinueStemnt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_ContinueStemnt(
  const Statement &statement,
  exprt &dest)
{
  dest.set("statement", "continue");
}

/*******************************************************************\

Function: convert_ct::convert_BreakStemnt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_BreakStemnt(
  const Statement &statement, 
  exprt &dest)
{
  dest.set("statement", "break");
}

/*******************************************************************\

Function: convert_ct::convert_GotoStemnt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_GotoStemnt(
  const GotoStemnt &statement,
  exprt &dest)
{
  dest.set("statement", "goto");

  if(statement.dest==NULL)
  {
    err_location(statement);
    throw "error: goto without destination";
  }

  dest.set("destination", statement.dest->name);
}

/*******************************************************************\

Function: convert_ct::convert_IfStemnt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_IfStemnt(
  const IfStemnt &statement,
  exprt &dest)
{
  dest.set("statement", "ifthenelse");

  dest.reserve_operands(3);
  
  dest.operands().push_back((exprt &)nil_rep);
  if(statement.cond!=NULL)
  {
    exprt &cond=dest.operands().back();
    convert(*statement.cond, cond);
  }

  dest.operands().push_back((exprt &)nil_rep);
  if(statement.thenBlk!=NULL)
    convert_Statement(*statement.thenBlk, dest.operands().back());

  dest.operands().push_back((exprt &)nil_rep);
  if(statement.elseBlk!=NULL)
    convert_Statement(*statement.elseBlk, dest.operands().back());
}

/*******************************************************************\

Function: convert_ct::convert_SwitchStemnt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_SwitchStemnt(
  const SwitchStemnt &statement,
  exprt &dest)
{
  dest.set("statement", "switch");

  dest.operands().resize(2);

  convert(*statement.cond,  dest.op0());

  convert_Statement(*statement.block, dest.op1());
}

/*******************************************************************\

Function: convert_ct::convert_Block

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_Block(
  const Block &block,
  exprt &dest)
{
  dest.set("statement", "block");

  // count it - for performance

  unsigned count=0;
  for(const Statement *stemnt=block.head; stemnt; stemnt=stemnt->next)
    count++;

  dest.reserve_operands(count);

  // now do the actual conversion  

  for(const Statement *stemnt=block.head; stemnt; stemnt=stemnt->next)
  {
    exprt code;
    convert_Statement(*stemnt, code);
    dest.move_to_operands(code);
  }
}

/*******************************************************************\

Function: convert_ct::convert_labels

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_labels(
  const Statement &statement,
  exprt &dest)
{
  if(statement.labels.empty())
    return;

  exprt op;
  op.swap(dest);

  exprt *last=NULL;

  for(LabelVector::const_iterator it=statement.labels.begin();
      it!=statement.labels.end(); it++)
  {
    exprt label_statement("code");
    label_statement.set("statement", "label");
    set_location(label_statement, statement.location);
   
    switch((*it)->type)
    {
    case LT_Goto:
      if((*it)->name==NULL)
      {
        err_location(statement);
        throw "goto label without name";
      }

      if(has_prefix((*it)->name->name, "CPROVER_ASYNC_"))
        label_statement.set("statement", "start_thread");
      else
        label_statement.set("label", (*it)->name->name);
      
      break;

    case LT_Default: // default:
      label_statement.set("default", true);
      break;

    case LT_Case: // case x:
      {
        exprt tmp;
        convert(*(*it)->begin, tmp);
        static_cast<exprt &>(label_statement.add("case")).move_to_operands(tmp);
      }
      break;

    default:
      err_location(statement);
      err << std::endl << statement << std::endl;
      throw "unknown label";
    }
    
    if(last==NULL)
    {
      dest.swap(label_statement);
      last=&dest;
    }
    else
    {
      last->move_to_operands(label_statement);
      last=&last->op0();
    }
  }

  assert(last!=NULL);
  last->move_to_operands(op);
}

/*******************************************************************\

Function: convert_ct::convert_Statement

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_Statement(
  const Statement &statement, 
  exprt &dest)
{
  dest.id("code");

  set_location(dest, statement.location);

  switch(statement.type)
  {
  case ST_NullStemnt:
    dest.set("statement", "skip");
    break;

  case ST_DeclStemnt:       convert_DeclStemnt((const DeclStemnt &)statement, dest); break;
  case ST_TypedefStemnt:    convert_DeclStemnt((const TypedefStemnt &)statement, dest); break;
  case ST_ExpressionStemnt: convert_ExpressionStemnt((const ExpressionStemnt &)statement, dest); break;
  case ST_IfStemnt:         convert_IfStemnt((const IfStemnt &)statement, dest); break;
  case ST_SwitchStemnt:     convert_SwitchStemnt((const SwitchStemnt &)statement, dest); break;
  case ST_ForStemnt:        convert_ForStemnt((const ForStemnt &)statement, dest); break;
  case ST_WhileStemnt:      convert_WhileStemnt((const WhileStemnt &)statement, dest); break;
  case ST_DoWhileStemnt:    convert_DoWhileStemnt((const DoWhileStemnt &)statement, dest); break;
  case ST_ContinueStemnt:   convert_ContinueStemnt(statement, dest); break;
  case ST_BreakStemnt:      convert_BreakStemnt(statement, dest); break;
  case ST_GotoStemnt:       convert_GotoStemnt((const GotoStemnt &)statement, dest); break;
  case ST_ReturnStemnt:     convert_ReturnStemnt((const ReturnStemnt &)statement, dest); break;
  case ST_Block:            convert_Block((const Block &)statement, dest); break;

  case ST_FileLineStemnt:
    // ignored: #line ...
    break;

  case ST_InclStemnt:
  case ST_EndInclStemnt:
    err_location(statement);
    err << "Warning: statement ignored: " << statement;
    warning();
    break;

  default:
    err_location(statement);
    err << "error: unknown statement: " << statement;
    throw 0;
  }

  convert_labels(statement, dest);
}
