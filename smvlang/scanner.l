%{
#include <stdio.h>
#include <string.h>

#include <expr.h>

#define PARSER smv_parser
#define YYSTYPE unsigned

#include "smv_parser.h"
#include "y.tab.h"

int yysmverror(const std::string &error);

void newlocation(YYSTYPE &x)
{
  newstack(x);
  PARSER.set_location(stack(x));
}

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
			  yysmverror("EOF in comment");
			  break;
                        }
                    }
                }

"//".*          ;
"--".*          ;
"#".*		{ /* preprocessor */ }

[ \t\n]+	;

"INIT"		{ newlocation(yysmvlval); return INIT_Token; }
"TRANS"		{ newlocation(yysmvlval); return TRANS_Token; }
"SPEC"		{ newlocation(yysmvlval); return SPEC_Token; }
"INVAR"		{ newlocation(yysmvlval); return INVAR_Token; }
"VAR"		{ newlocation(yysmvlval); return VAR_Token; }
"ASSIGN"	{ newlocation(yysmvlval); return ASSIGN_Token; }
"DEFINE"	{ newlocation(yysmvlval); return DEFINE_Token; }
"FAIRNESS"	{ newlocation(yysmvlval); return FAIRNESS_Token; }
"MODULE"	{ newlocation(yysmvlval); return MODULE_Token; }
"EXTERN"	{ newlocation(yysmvlval); return EXTERN_Token; }

"AG"		return AG_Token;
"AF"		return AF_Token;
"AX"		return AX_Token;
"next"		return NEXT_Token;
"init"		return init_Token;
"case"		return CASE_Token;
"esac"		return ESAC_Token;
"switch"	return SWITCH_Token;
"array"		return ARRAY_Token;
"of"		return OF_Token;
"boolean"	return BOOLEAN_Token;
"union"		return UNION_Token;
".."		return DOTDOT_Token;
"XOR"		return XOR_Token;

[\$A-Za-z_][A-Za-z0-9_\$#-]*	{
                  newstack(yysmvlval);
                  stack(yysmvlval).id(yytext);
		  return STRING_Token;
		}
\\[A-Za-z0-9_\$#-]*	{
                  newstack(yysmvlval);
                  stack(yysmvlval).id(yytext);
		  return QSTRING_Token;
		}
[0-9][0-9]*	{
                  newstack(yysmvlval);
                  stack(yysmvlval).id(yytext);
		  return NUMBER_Token;
		}
\"[^\"]*\"      {
                  newstack(yysmvlval);
                  std::string tmp(yytext);
                  stack(yysmvlval).id(std::string(tmp, 1, tmp.size()-2));
                  return QUOTE_Token;
                }

"~"		return NOT_Token;
"!"		return NOT_Token;
"+"		return PLUS_Token;
"|"		return OR_Token;
"&"		return AND_Token;
"->"		return IMPLIES_Token;
"<->"		return EQUIV_Token;
"="		return EQUAL_Token;
"<="		return LE_Token;
">="		return GE_Token;
"<"		return LT_Token;
">"		return GT_Token;
"!="		return NOTEQUAL_Token;
":="		return BECOMES_Token;
"."		return DOT_Token;

.		return yytext[0];

%%

int yywrap() { return 1; }
