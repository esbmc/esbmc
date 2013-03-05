%{
  // A parser for smtlib responses

#include "smtlib_conv.h"

#include "y.tab.hpp"

int smtliblex();
int smtliberror(const std::string &error);

%}

/* Values */
%union {
  const char *text;
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

/* Start token, for the response */
%start response

/* Types */

%%

/* Rules */

response: s_expr

spec_constant: TOK_NUMERAL | TOK_DECIMAL | TOK_HEXNUM | TOK_BINNUM |
               TOK_STRINGLIT

symbol: TOK_SIMPLESYM | TOK_QUOTEDSYM

numlist: TOK_NUMERAL | numlist TOK_NUMERAL

identifier: symbol | TOK_LPAREN TOK_KW_USCORE symbol numlist TOK_RPAREN

sexpr_list: | s_expr sexpr_list

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

info_response_arg: error_behaviour | TOK_STRINGLIT | reason_unknown

info_response: attribute | TOK_KEYWORD info_response_arg

info_response_list: info_response | info_response_list info_response

get_info_response: TOK_LPAREN info_response_list TOK_RPAREN

check_sat_response:

get_assertions_response:

get_proof_response:

get_unsat_core_response:

get_value_response:

get_assignment_response:
