%lex-param { int startsym }
%parse-param { int startsym }
%{
  // A parser for smtlib responses

#include <string>

#include "smtlib_conv.h"

#include "y.tab.hpp"

int smtliblex(int startsym);
int smtliberror(int startsym, const std::string &error);
%}

/* Values */
%union {
  char *text;
  struct sexpr *expr;
  std::string *str;
};

/* Some tokens */
%token <text> TOK_LPAREN
%token <text> TOK_RPAREN
%token <text> TOK_NUMERAL
%token <text> TOK_DECIMAL
%token <text> TOK_HEXNUM
%token <text> TOK_BINNUM
%token <text> TOK_STRINGLIT
%token <text> TOK_SIMPLESYM
%token <text> TOK_QUOTEDSYM
%token <text> TOK_KEYWORD
%token <text> TOK_KW_PAR
%token <text> TOK_KW_NUMERAL
%token <text> TOK_KW_DECIMAL
%token <text> TOK_KW_STRING
%token <text> TOK_KW_USCORE
%token <text> TOK_KW_EXCL
%token <text> TOK_KW_AS
%token <text> TOK_KW_LET
%token <text> TOK_KW_FORALL
%token <text> TOK_KW_EXISTS
%token <text> TOK_KW_UNSUPPORTED
%token <text> TOK_KW_SUCCESS
%token <text> TOK_KW_ERROR
%token <text> TOK_KW_IMMEXIT
%token <text> TOK_KW_CONEXECUTION
%token <text> TOK_KW_MEMOUT
%token <text> TOK_KW_INCOMPLETE
%token <text> TOK_KW_SAT
%token <text> TOK_KW_UNSAT
%token <text> TOK_KW_UNKNOWN
%token <text> TOK_KW_TRUE
%token <text> TOK_KW_FALSE

%token <text> TOK_START_GEN
%token <text> TOK_START_INFO
%token <text> TOK_START_SAT
%token <text> TOK_START_ASSERTS
%token <text> TOK_START_UNSATS
%token <text> TOK_START_VALUE
%token <text> TOK_START_ASSIGN
%token <text> TOK_START_OPTION

/* Start token, for the response */
%start response

/* Types */

%type <str> spec_constant

%%

/* Rules */

response: TOK_START_GEN gen_response |
          TOK_START_INFO get_info_response
          { yychar = YYEOF; }
          | TOK_START_SAT check_sat_response |
          TOK_START_ASSERTS get_assertions_response |
          TOK_START_UNSATS get_unsat_core_response |
          TOK_START_VALUE get_value_response |
          TOK_START_ASSIGN get_assignment_response |
          TOK_START_OPTION get_option_response

spec_constant: TOK_NUMERAL { $$ = new std::string($1); free($1); }
               | TOK_DECIMAL { $$ = new std::string($1); free($1); }
               | TOK_HEXNUM { $$ = new std::string($1); free($1); }
               | TOK_BINNUM { $$ = new std::string($1); free($1); }
               | TOK_STRINGLIT { $$ = new std::string($1); free($1); }

symbol: TOK_SIMPLESYM | TOK_QUOTEDSYM

symbol_list_empt: | symbol_list_empt symbol

numlist: TOK_NUMERAL | numlist TOK_NUMERAL

identifier: symbol | TOK_LPAREN TOK_KW_USCORE symbol numlist TOK_RPAREN

sexpr_list: | sexpr_list s_expr

s_expr: spec_constant | symbol | TOK_KEYWORD | TOK_LPAREN sexpr_list TOK_RPAREN

attribute_value: spec_constant | symbol | TOK_LPAREN sexpr_list TOK_RPAREN

attribute: TOK_KEYWORD | TOK_KEYWORD attribute_value

attr_list: attribute | attr_list attribute

sort_list: sort | sort_list sort

sort: identifier | TOK_LPAREN identifier sort_list TOK_RPAREN

qual_identifier: identifier | TOK_LPAREN TOK_KW_AS identifier sort TOK_RPAREN

var_binding: TOK_LPAREN symbol term TOK_RPAREN

varbind_list: var_binding | varbind_list var_binding

sorted_var: TOK_LPAREN symbol sort TOK_RPAREN

sortvar_list: sorted_var | sortvar_list sorted_var

term_list: term | term_list term

term_list_empt: | term | term_list term

term: spec_constant | qual_identifier | TOK_LPAREN qual_identifier TOK_RPAREN |
      TOK_LPAREN TOK_KW_LET TOK_LPAREN varbind_list TOK_RPAREN term TOK_RPAREN |
      TOK_LPAREN TOK_KW_FORALL TOK_LPAREN sortvar_list TOK_RPAREN term TOK_RPAREN |
      TOK_LPAREN TOK_KW_EXISTS TOK_LPAREN sortvar_list TOK_RPAREN term TOK_RPAREN |
      TOK_LPAREN TOK_KW_EXCL term attr_list TOK_RPAREN

gen_response: TOK_KW_UNSUPPORTED | TOK_KW_SUCCESS |
              TOK_LPAREN TOK_KW_ERROR TOK_STRINGLIT TOK_RPAREN

error_behaviour: TOK_KW_IMMEXIT | TOK_KW_CONEXECUTION

reason_unknown: TOK_KW_MEMOUT | TOK_KW_INCOMPLETE

status: TOK_KW_SAT | TOK_KW_UNSAT | TOK_KW_UNKNOWN

info_response_arg: error_behaviour | reason_unknown

info_response: attribute | TOK_KEYWORD info_response_arg

info_response_list: info_response | info_response_list info_response

get_info_response: TOK_LPAREN info_response_list TOK_RPAREN

check_sat_response: status

get_assertions_response: TOK_LPAREN term_list_empt TOK_RPAREN

/* get_proof_response: we're not going to be doing this */

get_unsat_core_response: TOK_LPAREN symbol_list_empt TOK_RPAREN

valuation_pair: TOK_LPAREN term term TOK_RPAREN

valuation_pair_list: valuation_pair | valuation_pair_list valuation_pair

get_value_response: TOK_LPAREN valuation_pair_list TOK_RPAREN

b_value: TOK_KW_TRUE | TOK_KW_FALSE

t_valuation_pair: TOK_LPAREN symbol b_value TOK_RPAREN

t_valuation_pair_empt: | t_valuation_pair_empt t_valuation_pair

get_assignment_response: TOK_LPAREN t_valuation_pair_empt TOK_RPAREN

get_option_response: attribute_value
