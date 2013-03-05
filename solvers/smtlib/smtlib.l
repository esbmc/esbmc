%option nounput
%option noyywrap

%{

// A scanner for lexing smtlib2 responses from SMT solvers

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

symtext       [~!@%^&*_-+=<>.?/a-zA-Z]
symchars      [0-9{symtext}]
simplesym     {symtext}{symchars}*
quotedsym     |[^\\]*|
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

%%
