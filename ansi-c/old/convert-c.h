/*******************************************************************\

Module: ANSI-C Conversion / Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CONVERT_C_H
#define CPROVER_CONVERT_C_H

#include <iostream>
#include <map>
#include <set>

#include <ctool/ctool.h>

#include <mp_arith.h>

#include "error_handler.h"

bool convert_c(
  Project &project,
  std::list<symbolt> &symbols,
  message_handlert &message_handler,
  const std::string &module);

bool convert_c(
  const Expression &expression,
  exprt &expr,
  message_handlert &message_handler,
  const std::string &module);

class convert_ct:public error_handlert
{
public:
  convert_ct(    
    std::list<symbolt> &_symbols,
    const std::string &_module,
    message_handlert &_message_handler):
    error_handlert(_message_handler),
    anon_struct_count(0),
    language_prefix("c::"),
    symbols(_symbols),
    scope_prefix(language_prefix),
    module(_module)
  { }
      
  void convert(const TransUnit &unit);
  void convert(const Expression &expression, exprt &dest);

protected:
  typedef std::map<const SymEntry *, std::string> symbol_mapt;
  typedef std::map<const ScopeTbl *, unsigned> scope_mapt;
  scope_mapt scope_map;
  symbol_mapt symbol_map;

  std::string scope2string(const ScopeTbl *scope);

  void convert(const FunctionDef &functiondef);

  void convert(
    const Decl &decl,
    const Location &location,
    bool function_argument,
    bool global,
    symbolt &symbol);
               
  void convert(const DeclStemnt &statement);
  void convert(const Block &block);
  
  // Types
  void convert(const Type &type, const Location &location, typet &dest);
  void convert(const BaseType &type, const Location &location, typet &dest);
  void convert(const FunctionType &type, const Location &location, typet &dest);
  void convert(const PtrType &type, const Location &location, typet &dest);
  void convert(const BitFieldType &type, const Location &location, typet &dest);
  void convert(const ArrayType &type, const Location &location, typet &dest);
  void convert_UserType(const BaseType &type, const Location &location, typet &dest);
  void convert_StructType(const BaseType &type, const Location &location, typet &dest);
  void convert_StructType(const StructDef &structdef, const Location &location, typet &dest);

  // Expressions -- only does conversion, no type checking
  void convert(const CharConstant &expression, exprt &dest);
  void convert(const IntConstant &expression, exprt &dest);
  void convert(const UIntConstant &expression, exprt &dest);
  void convert(const FloatConstant &expression, exprt &dest);
  void convert(const StringConstant &expression, exprt &dest);
  void convert(const ArrayConstant &expression, exprt &dest);
  void convert(const EnumConstant &expression, exprt &dest);
  void convert(const Constant &expression, exprt &dest);
  void convert(const Variable &expression, exprt &dest);
  void convert(const FunctionCall &expression, exprt &dest);
  void convert(const UnaryExpr &expression, exprt &dest);
  void convert(const BinaryExpr &expression, exprt &dest);
  void convert(const TrinaryExpr &expression, exprt &dest);
  void convert(const IndexExpr &expression, exprt &dest);
  void convert(const SizeofExpr &expression, exprt &dest);
  void convert(const AssignExpr &expression, exprt &dest);
  void convert(const RelExpr &expression, exprt &dest);
  #ifdef HAS_QuantExpr
  void convert(const QuantExpr &expression, exprt &dest);
  #endif
  void convert(const CastExpr &expression, exprt &dest);
  void convert(const SymEntry *entry, const Location &location, irept &dest);
  void convert(const symbolt &symbol, const locationt &location, exprt &dest);
  void convert(const symbolt &symbol, const locationt &location, typet &dest);
  void member(const BinaryExpr &expression, exprt &dest);
  void ptrmember(const BinaryExpr &expression, exprt &dest);
  void convert_enum(const EnumDef &def,
                    const std::string &name,
                    mp_integer &offset);
  bool pointer_arithmetic(exprt &dest);

  // Convert Code
  void convert(const Statement &statement);  
  void convert_Decl(const Decl &decl, const Location &location,
                    exprt &dest, bool function_argument);
  void convert_DeclStemnt(const DeclStemnt &statement, exprt &dest);
  void convert_ExpressionStemnt(const ExpressionStemnt &statement, exprt &dest);
  void convert_Statement(const Statement &statement, exprt &dest);
  void convert_IfStemnt(const IfStemnt &statement, exprt &dest);
  void convert_SwitchStemnt(const SwitchStemnt &statement, exprt &dest);
  void convert_ForStemnt(const ForStemnt &statement, exprt &dest);
  void convert_WhileStemnt(const WhileStemnt &statement, exprt &dest);
  void convert_DoWhileStemnt(const DoWhileStemnt &statement, exprt &dest);
  void convert_ContinueStemnt(const Statement &statement, exprt &dest);
  void convert_BreakStemnt(const Statement &statement, exprt &dest);
  void convert_GotoStemnt(const GotoStemnt &statement, exprt &dest);
  void convert_ReturnStemnt(const ReturnStemnt &statement, exprt &dest);
  void convert_Block(const Block &block, exprt &dest);
  void convert_labels(const Statement &statement, exprt &dest);
  
  void err_location(const Location &location);
  
  void err_location(const Expression &expression)
  { err_location(expression.location); }
   
  void err_location(const Statement &statement)
  { err_location(statement.location); }
 
  void set_location(irept &dest, const Location &location) const;
  void convert_location(locationt &l, const Location &location) const;
   
  void print_symbolptr_list(
    std::ostream &out,
    const symbolptr_listt &symbolptr_list) const;
                 
  std::string function_name;
  std::string typedef_name;
  unsigned anon_struct_count;
  typet function_type; 
  std::string language_prefix;
  std::list<symbolt> &symbols;
  
  std::string scope_prefix;
  std::string module;
};

#endif
