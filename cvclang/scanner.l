%{
#define YYSTYPE exprt
#include "expr.h"
#include "y.tab.h"
#include "cvc_parser.h"
%}

%option noyywrap
%option nounput

w	[\ \t\n]*
s	[\ \t\n]
t	[\ \t]
ts      [\ \t]*
block   ("{-"([^-]*|("-"+[^}-]))*"-}")

%%

{ts}"\n"{w}     return NEWLINE;
{t}			/* eat up single blanks and tabs */
{w}"--".*$		/* eat up comments */
{w}{block}{w}		/* eat up block comments */
{w}"transparent".*$	/* eat up transparent-declarations from FDR-2-files. */
    
{w}"assert"{w}	return ASSERT;
{w}"["{w}	return LSQUARE;
{w}"]"{w}	return RSQUARE;
{w}"[["{w}	return LSUBST;
{w}"]]"{w}	return RSUBST;
{w}"{"{w}	return LBRACE;
{w}"}"	        return RBRACE;
{w}"{|"{w}	return LPBRACE;
{w}"|}" 	return RPBRACE;
"("{w}		return OPEN;
{w}")"		return CLOSE;
"<"{w}		return LT;
{w}">"		return GT;
{w}"<="{w}	return LE;
{w}">="{w}	return GE;
{w}"[|"{w}	return LCOMM;
{w}"|]"{w}	return RCOMM;
{w}"||"{w}	return PAR;
{w}"[]"{w}	return BOX;
{w}"|~|"{w}	return NDET;
{w}"|||"{w}	return INTL;
{w}"|"{w}	return PIPE;
{w}"\\"{w}	return BACKSLASH;
{w}"->"{w}	return ARROW;
{w}"<-"{w}	return BECOMES;
{w}":"{w}	return COLON;
{w}"/\\"{w}	return INTR;
{w}";"{w}	return SEMI;
{w}"."{w}	return DOT;
{w}"?"{w}	return QUERY;
{w}"!"{w}	return PLING;
{w}"@"{w}	return AT;
{w}","{w}	return COMMA;
{w}"="{w}	return EQUAL;
{w}"+"{w}	return PLUS;
{w}"-"{w}	return MINUS;
{w}"*"{w}	return TIMES;
{w}"/"{w}	return SLASH;
{w}"%"{w}	return MOD;
{w}"#"{w}	return HASH;
{w}"=="{w}	return EQ;
{w}"!="{w}	return NE;
{w}"^"{w}	return CAT;
{w}"&"{w}	return GUARD;
{w}".."{w}	return DOTDOT;

{w}"STOP"		return STOP;
{w}"SKIP"		return SKIP;
{w}"CHAOS"{w}   	return CHAOS;
{w}"true"		return CSPTRUE;
{w}"false"		return CSPFALSE;
"channel"{s}{w}		return CHANNEL;
"datatype"{s}{w}	return DATATYPE;
{w}"nametype"{s}{w}	return NAMETYPE;
{w}"union"{w}		return UNION;
{w}"inter"{w}		return INTER;
{w}"diff"{w}		return DIFF;
{w}"head"{w}		return HEAD;
{w}"tail"{w}		return TAIL;
{w}"not"{s}{w}		return NOT;
{w}"and"{s}{w}		return AND;
{w}"or"{s}{w}		return OR;
{w}"if"{s}{w}		return IF;
{w}"then"{s}{w}		return THEN;
{w}"else"{s}{w}		return ELSE;
{w}"let"{s}{w}		return LET;
{w}"within"{s}{w}	return WITHIN;
{w}"lambda"{s}{w}	return LAMBDA;
{w}"ldot"{s}{w}		return LDOT;

[[:alpha:]][[:alnum:]_]*[']*   { yycvclval.id(yytext); return NAME; }
[[:digit:]]+		       { yycvclval.id(yytext); return NUMBER; }

.               ;

%%
