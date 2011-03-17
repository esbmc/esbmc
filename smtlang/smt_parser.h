/*******************************************************************\
 *
 * Module: SMT-LIB Parser
 *
 * Author: CM Wintersteiger 
 *
\*******************************************************************/

#ifndef _SMT_PARSER_H_
#define _SMT_PARSER_H_

#include <parser.h>
#include <i2string.h>

#include "smt_parse_tree.h"

enum PARSERTOKENS { NOT=0, AND, OR, IMPLIES, XOR, IFF, IF_THEN_ELSE, 
  IDENT, NUMERAL, STATUS_ID, RATIONAL, INDEX, LPAR, RPAR, LBR, RBR, LCB, 
  RCB, STATUS, BENCHMARK, LOGICA, EXTRAFUNS, EXTRAPREDS, EXTRASORTS,
  ASSUMPTION, FORMULA, TRUE, FALSE, DISTINCT, LET, FLET, ITE, VAR, FVAR,
  EQ, LT, GT, PLUS, MINUS, MULT, DIV, MOD, ANOT, LEQ, GEQ,
  FORALL, EXISTS, ARITH_SYMB, NOTES, QUOTE, STRING_CONTENT, ATTRIBUTE,
  USER_VALUE_CONTENT, THEORY, THEORYA, SORTS, FUNS, PREDS, AXIOMS,
  DEFINITION, LANGUAGE, EXTENSIONS, LOGIC, LEXEOF };
  
extern const char* token_names[];

class smt_parsert : public parsert
{
  private:
    unsigned parenthesis_counter;
    int last_token, lookahead_token;
    exprt last_value, lookahead_value;
    unsigned last_line_no, last_col_no;
     
    int get_token( void );
    bool expect( enum PARSERTOKENS );
    
    bool parse_theory( void );
    bool parse_logic( void );
    bool parse_benchmark( void );
    
    bool parse_formula( exprt&, bool prereadLPAR=false);
    bool parse_term( exprt& );
    bool parse_baseterm( exprt& e );
    bool parse_annotations( void );
    bool parse_prop_atom( exprt& );
    bool parse_connective( exprt& );
    bool parse_atom( exprt& );
    bool parse_simple_ident( exprt& );
    bool parse_ident( exprt& );
    bool parse_extrasorts( smt_parse_treet::benchmarkt::sort_symbolst& );
    bool parse_extrafuns( smt_parse_treet::benchmarkt::function_symbolst& );
    bool parse_extrapreds( smt_parse_treet::benchmarkt::predicate_symbolst& );
    bool parse_quant_var( exprt& );
    bool parse_quantification( exprt& );
    bool parse_string( void );
        
  public:
    smt_parse_treet parse_tree;
    unsigned column_no;
    
    virtual bool parse();
      
    virtual void clear()
    {
      column_no=0;
      parsert::clear();
      parse_tree.clear();
    }
    
    void set_location(exprt &e)
    {
      locationt &l=e.location();
  
      l.set_line(line_no);
      l.set_column(column_no);
  
      if(filename!="")
        l.set_file(filename);
    }
    
    bool parse_error(const std::string &message);    
};

extern smt_parsert smt_parser;
#define PARSER smt_parser

#endif /* _SMT_PARSER_H_ */
