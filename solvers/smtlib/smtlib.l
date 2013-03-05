%option nounput
%option noyywrap

%{

// A scanner for lexing smtlib2 responses from SMT solvers


#include "y.tab.hpp"
%}

%{
  // Basic lexicon
%}

lparen        \(
rparen        \)
numeral       (0|[1-9][0-9]*)
decimal       {numeral}.0*{numeral}
hexnum        #x[a-fA-F0-9]+
binnum        #b[01]+
escapequote   \\"
escapeslash   \\\\
stringlit     "((.|{escapequote}|{escapeslash})*)"

symtext       [-~!@%^&*_+=<>.?/a-zA-Z]
symchars      [0-9{symtext}]
simplesym     {symtext}{symchars}*
quotedsym     \|[^\\]*\|
keyword       :{symchars}+

%%

%{
  // Keywords -- we don't particularly care about using them when reading.
%}

"par"                 { return TOK_KW_PAR; }
"NUMERAL"             { return TOK_KW_NUMERAL; }
"DECIMAL"             { return TOK_KW_DECIMAL; }
"STRING"              { return TOK_KW_STRING; }
"_"                   { return TOK_KW_USCORE; }
"!"                   { return TOK_KW_EXCL; }
"as"                  { return TOK_KW_AS; }
"let"                 { return TOK_KW_LET; }
"forall"              { return TOK_KW_FORALL; }
"exists"              { return TOK_KW_EXISTS; }
"unsupported"         { return TOK_KW_UNSUPPORTED; }
"success"             { return TOK_KW_SUCCESS; }
"error"               { return TOK_KW_ERROR; }
"immediate-exit"      { return TOK_KW_IMMEXIT; }
"continued-execution" { return TOK_KW_CONEXECUTION; }
"memout"              { return TOK_KW_MEMOUT; }
"incomplete"          { return TOK_KW_INCOMPLETE; }
"sat"                 { return TOK_KW_SAT; }
"unsat"               { return TOK_KW_UNSAT; }
"unknown"             { return TOK_KW_UNKNOWN; }
"true"                { return TOK_KW_TRUE; }
"false"               { return TOK_KW_FALSE; }

{lparen}              { return TOK_LPAREN; }
{rparen}              { return TOK_RPAREN; }
{numeral}             { smtliblval.text = strdup(yytext); return TOK_NUMERAL; }
{decimal}             { smtliblval.text = strdup(yytext); return TOK_DECIMAL; }
{hexnum}              { smtliblval.text = strdup(yytext); return TOK_HEXNUM; }
{binnum}              { smtliblval.text = strdup(yytext); return TOK_BINNUM; }
{stringlit}           { smtliblval.text = strdup(yytext); return TOK_STRINGLIT;}
{simplesym}           { smtliblval.text = strdup(yytext); return TOK_SIMPLESYM;}
{quotedsym}           { smtliblval.text = strdup(yytext); return TOK_QUOTEDSYM;}
{keyword}             { smtliblval.text = strdup(yytext); return TOK_KEYWORD;}

%%
