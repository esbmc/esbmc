%{
#ifdef _WIN32
#define YY_NO_UNISTD_H
#endif

#include <stdio.h>
#include <string.h>

#include <expr.h>

#define PARSER bp_parser
#define YYSTYPE unsigned

#include "bp_parser.h"
#include "y.tab.h"

int yybperror(const std::string &error);

%}

%pointer

%%

"/*"		{
		  int c;

		  for(;;)
		    {
		      while((c = yyinput()) != '*' && c != EOF)
			;

		      if(c == '*')
                        {
			  while((c = yyinput()) == '*')
                            ;

			  if( c == '/' ) break;
                        }

		      if (c == EOF)
                        {
			  yybperror("EOF in comment");
			  break;
                        }
                    }
                }

"//".*          ;

[ \t\n]+	;

"begin"		return BEGIN_Token;
"end"		return END_Token;
"decl"		return DECL_Token;
"void"		return VOID_Token;
"bool"		return BOOL_Token;
"goto"		return GOTO_Token;
"while"		return WHILE_Token;
"do"		return DO_Token;
"skip"		return SKIP_Token;
"od"		return OD_Token;
"enforce"	return ENFORCE_Token;
"abortif"	return ABORTIF_Token;
"start_thread"	return START_THREAD_Token;
"end_thread"	return END_THREAD_Token;
"sync"		return SYNC_Token;
"atomic_begin"	return ATOMIC_BEGIN_Token;
"atomic_end"	return ATOMIC_END_Token;
"constrain"	return CONSTRAIN_Token;
"dead"		return DEAD_Token;
"return"	return RETURN_Token;
"if"		return IF_Token;
"then"		return THEN_Token;
"else"		return ELSE_Token;
"fi"		return FI_Token;
"dfs"		return DFS_Token;

"&"		return AND_Token;
"&&"		return AND_Token;
"|"		return OR_Token;
"||"		return OR_Token;
"'"		return TICK_Token;
"=>"		return IMPLIES_Token;
"!="		return NEQ_Token;
"="		return EQ_Token;
"^"		return XOR_Token;

":="		return ASSIGN_Token;
"schoose"	return SCHOOSE_Token;

[\$A-Za-z_][A-Za-z0-9_\$#-]*	{
                  newstack(yybplval);
                  stack(yybplval).id(yytext);
		  return IDENTIFIER_Token;
		}
[0-9][0-9]*	{
                  newstack(yybplval);
                  stack(yybplval).id(yytext);
		  return NUMBER_Token;
		}

.		return yytext[0];

%%

int yywrap() { return 1; }
