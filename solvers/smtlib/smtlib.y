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

/* Start token, for the response */
%start response

/* Types */

%%

/* Rules */

response:
