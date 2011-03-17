/*******************************************************************\
  
   Module: SMT-LIB Parser
  
   Author: CM Wintersteiger
  
\*******************************************************************/

#include <stdio.h>

#include "smt_parser.h"
#include "expr2smt.h"

//#define __DEBUGOUTPUT

smt_parsert smt_parser;

extern FILE* yysmtin;
extern int yysmtparse(void);

const char* token_names[] = { "not", "and", "or", "implication", "xor", "iff",
  "if_then_else", "identifier", "numeral", "status value", "rational", "index",
  "left parenthesis", "right parenthesis", "left bracket", "right bracket",
  "left curly bracket", "right curly bracket", "status attribute", 
  "benchmark definition", "logic attribute", "local functions", 
  "local predicates", "local sorts", "assumption attribute", "formula", 
  "true", "false", "distinct", "let", "flet", "ITE", "variable", 
  "function variable", "equality", "<", ">", "+", "-", "*", "/", "modulo", 
  "~", "<=", ">=", "forall", "exists", "arithmetic symbol", "notes attribute", 
  "quote", "string", "attribute", "user defined value", "theory definition",
  "theory attribute", "sorts attribute", "functions attribute", 
  "predicates attribute", "axioms attribute", "definition", 
  "language attribute", "extensions attribute", "logic definition", 
  "end-of-file" };  
  
/*******************************************************************\

Function: smt_parsert::parse_error

  Inputs: error message and next token description

 Outputs: nothing

 Purpose: outputs a parser error message

\*******************************************************************/

bool smt_parsert::parse_error( 
  const std::string &message)
{
  locationt location;
  location.set_file(filename);
  location.set_line(i2string(last_line_no));
  location.set_column(i2string(last_col_no));
  std::string tmp=message;
  print(1, tmp, -1, location);
  return true;
}

/*******************************************************************\

Function: smt_parsert::get_token

  Inputs: none

 Outputs: token kind

 Purpose: asks the lexer for the next token

\*******************************************************************/

int yysmtlex();
extern char yysmttext[];

int smt_parsert::get_token( void )
{
  last_token = lookahead_token;
  last_value = lookahead_value;
  last_line_no = line_no;
  last_col_no = column_no;
  
  lookahead_token = yysmtlex();
  if (stack.size()>0) lookahead_value = stack.back();
  
  #ifdef __DEBUGOUTPUT
  std::cout << "READ " << token_names[last_token] << std::endl;
  #endif
  
  if (last_token==LPAR) parenthesis_counter++;
  else if (last_token==RPAR) parenthesis_counter--;
  return last_token;
}

/*******************************************************************\

Function: smt_parsert::expect

  Inputs: token kind

 Outputs: true/false

 Purpose: 

\*******************************************************************/

bool smt_parsert::expect( enum PARSERTOKENS tok )
{
  if (get_token()!=tok)
    return parse_error( 
      std::string("Expected `") + token_names[tok] + "', got `" + 
      token_names[last_token] + "'");
  return true;
}

/*******************************************************************\

Function: smt_parsert::parse

  Inputs: none

 Outputs: nothing

 Purpose: parses an input file

\*******************************************************************/

bool smt_parsert::parse( void )
{  
  FILE* infile = fopen(filename.as_string().c_str(), "r");
  yysmtin = infile; 
  
  parenthesis_counter=0;
  get_token(); // prepare lookahead
  
  while (lookahead_token==LPAR)
  {
    get_token();
    switch(lookahead_token)
    {
      case BENCHMARK:
      {
        if (parse_benchmark()) return true;
        break;
      }
      case THEORY:
        if (parse_theory()) return true;
        break;
      case LOGIC:
        if (parse_logic()) return true;
        break;
      default:
        return parse_error(std::string("Toplevel: Unexpected ") + 
                           token_names[lookahead_token]);
    }
      
    if (!expect(RPAR)) return true;
  }
  
  if (parenthesis_counter!=0)
      return parse_error("Some parenthesis are missing.");
  
  if (!expect(LEXEOF)) return true;
  
  return false;
}

/*******************************************************************\

Function: smt_parsert::parse_theory

  Inputs: none

 Outputs: nothing

 Purpose: parses an input file

\*******************************************************************/

bool smt_parsert::parse_theory( void )
{
  parse_error("Theory files not supported");
  return true;
}

/*******************************************************************\

Function: smt_parsert::parse_logic

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_parsert::parse_logic( void )
{  
  parse_error("Logic files not supported");
  return true;
}

/*******************************************************************\

Function: smt_parsert::parse_benchmark

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_parsert::parse_benchmark( void )
{
  if (!expect(BENCHMARK)) return true;
  if (!expect(IDENT)) return true;
  
  parse_tree.benchmarks.resize(parse_tree.benchmarks.size()+1);  
  smt_parse_treet::benchmarkt &benchmark = parse_tree.benchmarks.back();
  benchmark.name = last_value; 
  
  while(lookahead_token!=EOF && lookahead_token!=RPAR)
  {
    // std::cout << "LA = " << token_names[lookahead_token] << std::endl;
    switch(lookahead_token)
    {
      case LOGICA:
      {
        get_token();
        benchmark.logics.push_back(exprt());
        if (parse_ident(static_cast<exprt&>(benchmark.logics.back()))) 
          return true;
        break;
      }
      case ASSUMPTION:
      {
        get_token();
        benchmark.assumptions.push_back(exprt());
        if (parse_formula(benchmark.assumptions.back())) return true;        
        break;
      }
      case FORMULA:
      {            
        get_token();
        benchmark.formulas.push_back(exprt());
        if (parse_formula(benchmark.formulas.back())) return true;        
        break;
      }
      case STATUS:    
      {    
        get_token();
        if (!expect(STATUS_ID)) return true;        
        if (last_value.id()=="sat")
          benchmark.status.push_back(smt_parse_treet::benchmarkt::SAT);
        else if (last_value.id()=="unsat")
          benchmark.status.push_back(smt_parse_treet::benchmarkt::UNSAT);
        else if (last_value.id()=="unknown")
          benchmark.status.push_back(smt_parse_treet::benchmarkt::UNKNOWN);
        else
          return parse_error(std::string("Invalid status tag: ") + 
                              last_value.id_string());
        break;
      }        
      case EXTRASORTS:
      {
        get_token();
        if (parse_extrasorts(benchmark.sort_symbols)) return true;
        break;
      }
      case EXTRAFUNS:
      {
        get_token();        
        if (parse_extrafuns(benchmark.function_symbols)) return true;        
        break;
      }
      case EXTRAPREDS:
      {
        get_token();
        if (parse_extrapreds(benchmark.predicate_symbols)) return true;
        break;
      }
      case NOTES:
      {
        get_token();
        if (parse_string()) return true;        
        break;
      }
      case ATTRIBUTE:
      {
        parse_annotations();
        break;
      }
      default:
        return parse_error(std::string("Unexpected ") + 
                           token_names[last_token]);
    }
  }
  
  return false;
}

/*******************************************************************\

Function: smt_parsert::parse_formula

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_parsert::parse_formula( exprt& e, bool prereadLPAR )
{
  #ifdef __DEBUGOUTPUT
    std::cout << "FORMULA" << std::endl;
  #endif
  
  unsigned oldlvl = parenthesis_counter;
  if (prereadLPAR) oldlvl--;
  
  while (lookahead_token==LPAR)
    get_token();
  
  switch(lookahead_token)
  {
    case NOT: case IMPLIES: case IF_THEN_ELSE: 
    case AND: case OR: case XOR: case IFF:
    {
      if (parse_connective(e)) return true;      
      while (lookahead_token!=RPAR && 
             lookahead_token!=ATTRIBUTE)
      {
        e.reserve_operands(e.operands().size()+1);
        e.operands().push_back(exprt());
        if (parse_formula(e.operands().back())) return true;
      } 
      parse_annotations();
      break;
    }    
    case LET:
    {
      get_token();
      if (!expect(LPAR)) return true;
      if (!expect(VAR)) return true;
      e.id("let");
      exprt &vars = static_cast<exprt&>(e.add("variables"));        
      exprt &terms = static_cast<exprt&>(e.add("vterms"));
      exprt var("var");
      var.set("identifier", last_value.id_string().substr(1));
      exprt vterm;      
      while (lookahead_token!=RPAR)
      {
        if (parse_term(vterm)) return true;
      }
      if (!expect(RPAR)) return true;
      exprt temp;
      if (parse_formula(temp)) return true;
      if (temp.id()=="let")
      {
        exprt &v = static_cast<exprt&>(temp.add("variables"));
        exprt &t = static_cast<exprt&>(temp.add("vterms"));
        vars.swap(v);
        terms.swap(t);
        e.move_to_operands(temp.op0());
      }
      else
      {
        e.move_to_operands(temp);
      }      
      // vars/terms will be reversed!
      terms.move_to_operands(vterm);
      vars.move_to_operands(var);
      parse_annotations();
      break;
    }
    case FLET:
    {
      get_token();
      if (!expect(LPAR)) return true;
      if (!expect(FVAR)) return true;
      e.id("flet");
      exprt &vars = static_cast<exprt&>(e.add("variables"));        
      exprt &formulas = static_cast<exprt&>(e.add("vformulas"));
      exprt var("fvar");
      var.set("identifier", last_value.id_string().substr(1));
      exprt vformula;      
      while (lookahead_token!=RPAR)
      {
        if (parse_formula(vformula)) return true;
      }
      if (!expect(RPAR)) return true;
      exprt temp;
      if (parse_formula(temp)) return true;
      if (temp.id()=="flet")
      {
        exprt &v = static_cast<exprt&>(temp.add("variables"));
        exprt &f = static_cast<exprt&>(temp.add("vformulas"));
        vars.swap(v);
        formulas.swap(f);
        e.move_to_operands(temp.op0());
      }
      else
      {
        e.move_to_operands(temp);
      }      
      // vars/formulas will be reversed!
      formulas.move_to_operands(vformula);
      vars.move_to_operands(var);
      parse_annotations();
      break;
    }
    case FORALL:
    case EXISTS:
    {
      if (parse_quantification(e)) return true;
      break;
    }
    case IDENT:
    {
      if (oldlvl==parenthesis_counter)
      { // no ()'s seen
        if (parse_ident(e)) return true;
      }
      else 
      { // this could be a function call, etc.
        if (parse_atom(e)) return true;
      }
      break;
    }
    case TRUE: case FALSE:
    case FVAR: case ITE:
    case ARITH_SYMB: case DISTINCT:    
    {      
      if (parse_atom(e)) return true;
      break;
    }    
    default:
      return parse_error(std::string("Formula: unexpected ") + 
                         token_names[lookahead_token]);
  }
  
  while (parenthesis_counter>oldlvl) 
    if (!expect(RPAR)) return true;
    
  if (parenthesis_counter>oldlvl)
    return parse_error(std::string("Expected `)', got `") + 
                       token_names[lookahead_token] + "'");
  
  #ifdef __DEBUGOUTPUT
    std::cout << "ENDFORMULA" << std::endl;
  #endif
  
  return false;
}

/*******************************************************************\

Function: smt_parsert::parse_annotations

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_parsert::parse_annotations( void )
{
  while (lookahead_token==ATTRIBUTE)
  {
    get_token();    
    if (lookahead_token==LCB)
    {
      get_token();
      if (lookahead_token==USER_VALUE_CONTENT) 
        get_token(); // skip it
      if (!expect(RCB)) return true;
    }
  }
  return false;
}

/*******************************************************************\

Function: smt_parsert::parse_prop_atom

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_parsert::parse_prop_atom( exprt& expr )
{
  switch(lookahead_token)
  {
    case TRUE:
      get_token(); 
      expr.id("bool");
      expr.set("value", "true");
      break;
    case FALSE:
      get_token(); 
      expr.id("bool");
      expr.set("value", "false");
      break;
    case FVAR:
      get_token(); 
      expr.id("fvar");
      expr.set("identifier", last_value.id_string().substr(1));
      break;
    case IDENT:
      if (parse_ident(expr)) return true;
      break;
    default:
      return parse_error("Expected propositional atom.");
  }
  return false;
}

/*******************************************************************\

Function: smt_parsert::parse_connective

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_parsert::parse_connective( exprt& expr )
{
  switch(lookahead_token)
  {
    case NOT: get_token(); expr.id("not"); break;
    case IMPLIES: get_token(); expr.id("impl"); break;
    case IF_THEN_ELSE: get_token(); expr.id("if_then_else"); break;
    case AND: get_token(); expr.id("and"); break;
    case OR: get_token(); expr.id("or"); break;
    case XOR: get_token(); expr.id("xor"); break;
    case IFF: get_token(); expr.id("iff"); break;
    default:
      return parse_error("Expected connective");
  }
  return false;
}

/*******************************************************************\

Function: smt_parsert::parse_atom

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_parsert::parse_atom( exprt& e )
{
  #ifdef __DEBUGOUTPUT
    std::cout << "ATOM" << std::endl;
  #endif
  
  bool musthaverpar = false;
  if (lookahead_token==LPAR) musthaverpar = true;
    
  switch(lookahead_token)
  {
    case IDENT: case ARITH_SYMB:
    case DISTINCT:
    {
      bool musthaveterms = false;
      switch(lookahead_token)
      {
        case IDENT:
          if (parse_ident(e)) return true;
          break;
        case ARITH_SYMB:
          get_token();        
          e = last_value;
          musthaveterms=true;
          break;
        case DISTINCT:
          get_token();
          e = last_value;
          musthaveterms=true;
          break;
        default:
          return parse_error("Expected ident, arithmetic symbol or `distinct'");
      }
      
      if (musthaveterms)
      { // this is a pred_symb, there must be at least one term.
        e.reserve_operands(e.operands().size()+1);      
        e.operands().push_back(exprt());
        if (lookahead_token==IDENT) // parse_term would be confused
        {
          if (parse_ident(e.operands().back())) return true;
        }
        else
        {
          if (parse_term(e.operands().back())) return true;
        }
      }
      // it could still be a pred_symb, so look for terms
      while (lookahead_token!=RPAR && lookahead_token!=ATTRIBUTE)
      {
        e.reserve_operands(e.operands().size()+1);
        e.operands().push_back(exprt());
        if (lookahead_token==IDENT) // parse_term would be confused
        {
          if (parse_ident(e.operands().back())) return true;
        }
        else
        {
          if (parse_term(e.operands().back())) return true;
        }
      }

      if (e.operands().size()>0) // it was a pred_symb!
        e.id(ii2string(e));

      parse_annotations();
      break;
    }
    case TRUE: case FALSE:
    case FVAR:
      if (parse_prop_atom(e)) return true;
      break;
    default:
      return parse_error("Expected an atom");
  }
  
  if (musthaverpar && !expect(RPAR)) return true;
  
  #ifdef __DEBUGOUTPUT
    std::cout << "ENDATOM" << std::endl;
  #endif
  return false;
}

/*******************************************************************\

Function: smt_parsert::parse_ident

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_parsert::parse_ident( exprt& e )
{
  if (parse_simple_ident(e)) return true;  
  if (lookahead_token==INDEX)
  {
    get_token();
    static_cast<exprt&>(e.add("index")).operands().push_back(last_value);
  }
  return false;
}

/*******************************************************************\

Function: smt_parsert::parse_simple_ident

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_parsert::parse_simple_ident( exprt& expr )
{
  if (!expect(IDENT)) return true;
  expr = last_value;
  return false;
}

/*******************************************************************\

Function: smt_parsert::parse_extrafuns

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_parsert::parse_extrafuns( 
  smt_parse_treet::benchmarkt::function_symbolst& fs )
{  
  if (!expect(LPAR)) return true;
  
  while (lookahead_token==LPAR)
  {
    get_token();
    
    exprt e;
    switch(lookahead_token)
    {
      case IDENT:
        if (parse_ident(e)) return true;
        break;
      case ARITH_SYMB:
        get_token();
        e = last_value;
        break;
      default:
        return parse_error(
          std::string("Expected identifier or arithmetic symbol, but got `") + 
          token_names[lookahead_token] + "'"); 
    }        
    e.reserve_operands(e.operands().size()+1);
    e.operands().push_back(exprt("sorts"));    
    while (lookahead_token==IDENT)
    {            
      exprt &sorts = e.operands()[0];
      exprt t;
      if (parse_ident(t)) return true;
      sorts.reserve_operands(sorts.operands().size()+1);
      sorts.operands().push_back(exprt("sort"));
      sorts.operands().back().type() = typet(ii2string(t));
    }
    if (!expect(RPAR)) return true;
    fs[e.id_string()].push_back(e);    
  }
  
  if (!expect(RPAR)) return true;
  return false;
}

/*******************************************************************\

Function: smt_parsert::parse_extrafuns

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_parsert::parse_extrasorts( 
  smt_parse_treet::benchmarkt::sort_symbolst& ss )
{ 
  if (!expect(LPAR)) return true;
  
  while (lookahead_token==IDENT)
  {
    exprt e;
    if (parse_ident(e)) return true;
    
    ss.push_back(exprt("sort"));
    ss.back().type() = typet(ii2string(e));
  }
  
  if (!expect(RPAR)) return true;
  return false;
}

/*******************************************************************\

Function: smt_parsert::parse_extrapreds

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_parsert::parse_extrapreds( 
  smt_parse_treet::benchmarkt::predicate_symbolst& ps )
{  
  if (!expect(LPAR)) return true;
  
  while (lookahead_token==LPAR)
  {
    get_token();
    
    exprt e;
    if (!expect(IDENT)) return true;
    e = last_value;
    e.reserve_operands(e.operands().size()+1);
    e.operands().push_back(exprt("sorts"));    
    while (lookahead_token==IDENT)
    {            
      exprt &sorts = e.operands()[0];
      exprt t;
      if (parse_ident(t)) return true;
      e.reserve_operands(e.operands().size()+1);
      sorts.operands().push_back(exprt("sort"));
      sorts.operands().back().type() = typet(ii2string(t));
    }
    if (!expect(RPAR)) return true;
    ps[e.id_string()].push_back(e);    
  }
  
  if (!expect(RPAR)) return true;
  return false;
}

/*******************************************************************\

Function: smt_parsert::parse_term

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_parsert::parse_term( exprt& e )
{ 
  #ifdef __DEBUGOUTPUT
    std::cout << "TERM" << std::endl;
  #endif
  
  if (lookahead_token==LPAR)
  {
    get_token();
    switch (lookahead_token)
    {
      case ITE:
      {
        get_token();
        e.id("ite");
        e.operands().resize(3);
        //std::cout << "ITE1" << std::endl;
        if (parse_formula(e.operands()[0])) return true;
        //std::cout << "ITE2" << std::endl;
        if (parse_term(e.operands()[1])) return true;
        //std::cout << "ITE3" << std::endl;
        if (parse_term(e.operands()[2])) return true;
        //std::cout << "ITEDONE" << std::endl;
        parse_annotations();
        break;
      }
      case IDENT:
      case ARITH_SYMB:
      {
        if (lookahead_token==ARITH_SYMB)
        {
          get_token();
          e = last_value;
          
          // this must be a fun_symb, there must be at least one term.
          e.reserve_operands(e.operands().size()+1);
          e.operands().push_back(exprt());
          if (parse_term(e.operands().back())) return true;  
        }
        else
        {
          if (parse_ident(e)) return true;
        }

        while (lookahead_token!=RPAR && lookahead_token!=ATTRIBUTE)
        {
          e.reserve_operands(e.operands().size()+1);
          e.operands().push_back(exprt());
          if (parse_term(e.operands().back())) return true;
        }
        
        if (e.operands().size()>0)
        { 
          e.id(ii2string(e));
          e.remove("index");          
        }
          
        parse_annotations();
        break;
      }
      default:
        return parse_error("Expected term");
    }
    if (!expect(RPAR)) return true;
  }
  else
    if (parse_baseterm(e)) return true;
  
  #ifdef __DEBUGOUTPUT
    std::cout << "ENDTERM" << std::endl;
  #endif
  return false;
}

/*******************************************************************\

Function: smt_parsert::parse_baseterm

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_parsert::parse_baseterm( exprt& e )
{
  #ifdef __DEBUGOUTPUT
    std::cout << "BASETERM" << std::endl;
  #endif
  
  switch(lookahead_token)
  {
    case VAR:
      get_token();
      e.id("var");
      e.set("identifier", last_value.id_string().substr(1));
      break;
    case NUMERAL:
      get_token();
      e = last_value;
      break;
    case RATIONAL:
      get_token();
      e = last_value;
      break;
    case IDENT:
      if (parse_ident(e)) return true;
      break;
    default:
      return parse_error(std::string("Expected baseterm, but got `") + 
                         token_names[lookahead_token] + "'");
  }
  
  #ifdef __DEBUGOUTPUT
    std::cout << "ENDBASETERM" << std::endl;
  #endif
  
  return false;
}

/*******************************************************************\

Function: smt_parsert::parse_quant_var

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_parsert::parse_quant_var( exprt& e )
{
  // if (!expect(LPAR)) return true; // we don't do this here 
                                     // (would have to unput it before)
  if (!expect(VAR)) return true;
  e.id("var");
  e.set("identifier", last_value.id_string().substr(1));
  e.type()=typet("sort");
  exprt t;
  if (parse_ident(t)) return true;
  e.type().set("type", ii2string(t));
  
  if (!expect(RPAR)) return true;
  return false;
}

/*******************************************************************\

Function: smt_parsert::parse_quantification

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_parsert::parse_quantification( exprt& e )
{
  switch(lookahead_token)
  {
    case FORALL:
      e.id("forall");
      break;
    case EXISTS:
      e.id("exists");
      break;
    default:
      return parse_error(std::string("Expected quantifier, but got `") + 
                         token_names[lookahead_token] + "'");
  }
  get_token();
  
  exprt& qv = e.add_expr("qvars");
  qv.id("qvars");
  
  // there must be at least one qvar
  qv.reserve_operands(qv.operands().size()+1);
  qv.operands().push_back(exprt());
  if (!expect(LPAR)) return true;
  parse_quant_var(qv.operands().back());
  
  while (lookahead_token==LPAR)
  {// this could be another qvar
    get_token();    
    if (lookahead_token==VAR)
    {
      qv.reserve_operands(qv.operands().size()+1);
      qv.operands().push_back(exprt());
      parse_quant_var(qv.operands().back());
    }
    else
    {       
      break; // go on with formula
    }
  }
  
  // the formula
  e.reserve_operands(e.operands().size()+1);
  e.operands().push_back(exprt());
  parse_formula(e.operands().back(), true);
  
  parse_annotations();
  return false;
}

/*******************************************************************\

Function: smt_parsert::parse_string

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_parsert::parse_string( void )
{
  if (!expect(QUOTE)) return true;
  if (!expect(STRING_CONTENT)) return true;
  if (!expect(QUOTE)) return true;
  return false;
}
