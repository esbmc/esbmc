%option nounput

%{

/*************** Includes and Defines *****************************/

#ifdef _WIN32
#define YY_NO_UNISTD_H
static int isatty(int) { return 0; }
#endif

#include <expr.h>
#include <arith_tools.h>
#include <config.h>

#include <ansi-c/c_types.h>
#include <ansi-c/convert_float_literal.h>
#include <ansi-c/convert_integer_literal.h>
#include <ansi-c/unescape_string.h>
#include <ansi-c/preprocessor_line.h>
#include <ansi-c/convert_character_literal.h>
#include <ansi-c/convert_string_literal.h>

#define PARSER cpp_parser
#define YYSTYPE unsigned
#undef  ECHO
#define ECHO

#include "cpp_parser.h"
#include "tokens.h"

void set_line_no();

%}

%x COMMENT1
%x COMMENT2
%x GCC_ATTRIBUTE
%x MSC_DECLSPEC
%x MSC_ASM

%{
void cpp_scanner_init()
{
  YY_FLUSH_BUFFER;
  BEGIN(0); // this is INITIAL, hopefully
}
%}

identifier [a-zA-Z_][0-9a-zA-Z_]*

exponent_part [eE][-+]?[0-9]+
fractional_constant ([0-9]*"."[0-9]+)|([0-9]+".")
floating_constant (({fractional_constant}{exponent_part}?)|([0-9]+{exponent_part}))[FfLl]?

integer_suffix [uUlL]
integer_suffix_opt {integer_suffix}*
decimal_constant [1-9][0-9]*{integer_suffix_opt}
octal_constant "0"[0-7]*{integer_suffix_opt}
hex_constant "0"[xX][0-9a-fA-F]+{integer_suffix_opt}

simple_escape [abfnrtv'"?\\]
octal_escape  [0-7]{1,3}
hex_escape "x"[0-9a-fA-F]+

escape_sequence [\\]({simple_escape}|{octal_escape}|{hex_escape})
c_char [^'\\\n]|{escape_sequence}
s_char [^"\\\n]|{escape_sequence}

h_tab [\011]
form_feed [\014]
v_tab [\013]
c_return [\015]

horizontal_white [ ]|{h_tab}

integer         [0-9]+
cppfile         "\""[ !#-~]*"\""
cpplineno       "#"{horizontal_white}*"line"*{horizontal_white}*{integer}.*"\n"
pragma          "#"{horizontal_white}*pragma{horizontal_white}.*"\n"

%%

%{
#define set_text() (PARSER.current_token().text=yytext,PARSER.set_location())
%}

<INITIAL>"/*"	{ BEGIN(COMMENT1); }	/* begin C comment state */

<COMMENT1>{
   "*/"		{ BEGIN(INITIAL); }	/* end comment state, back to INITIAL */
   "/*"		{ yycpperror("Probably nested comments"); }
   <<EOF>>	{ yycpperror("Unterminated comment"); }
   [^*/\n]*	{ /* ignore every char except '*' and NL (performance!) */ }
   .		{ } /* all single characters within comments are ignored */
   \n		{ } 
	}

<INITIAL>"//"	{ BEGIN(COMMENT2); }	/* begin C++ comment state */

<COMMENT2>{
   \n		{ BEGIN(INITIAL); }	/* end comment state, back to INITIAL */
   .*		{ } /* all characters within comments are ignored */
	}

<GCC_ATTRIBUTE>{
")"    { PARSER.parenthesis_counter--;
         if(PARSER.parenthesis_counter==0)
         BEGIN(INITIAL); }
"("    { PARSER.parenthesis_counter++; }
.      { /* Throw away */ }
}

<MSC_DECLSPEC>{
")"    { PARSER.parenthesis_counter--;
         if(PARSER.parenthesis_counter==0)
         BEGIN(INITIAL); }
"("    { PARSER.parenthesis_counter++; }
.      { /* Throw away */ }
}

<MSC_ASM>[^}]*  { /* Throw away */ }
<MSC_ASM>"}"    { BEGIN(INITIAL); }

<INITIAL>{
{horizontal_white}+
({v_tab}|{c_return}|{form_feed})+
({horizontal_white}|{v_tab}|{c_return}|{form_feed})*"\n"

__alignof__         { set_text(); return SIZEOF; }
__asm__             { set_text(); return ASM; }
__asm               { if(PARSER.mode==cpp_parsert::MSC)
                        BEGIN(MSC_ASM);
                      else
                      {
                        set_text();
                        return ASM;
                      }
                    }
__attribute         { BEGIN(GCC_ATTRIBUTE); PARSER.parenthesis_counter=0; }
__attribute__       { BEGIN(GCC_ATTRIBUTE); PARSER.parenthesis_counter=0; }
_cdecl		    { /* ignore */ }
__cdecl		    { /* ignore */ }
__const             { set_text(); return CONST; }
__declspec          { BEGIN(MSC_DECLSPEC); PARSER.parenthesis_counter=0; }
__extension__       { set_text(); return Ignore; }
_inline             { set_text(); return INLINE; }
__inline            { set_text(); return INLINE; }
__inline__          { set_text(); return INLINE; }
__noreturn__        { set_text(); return Ignore; }
restrict            { set_text(); return Ignore; }
__restrict__        { set_text(); return Ignore; }
__restrict          { set_text(); return Ignore; }
__stdcall           { set_text(); return Ignore; }
__signed            { set_text(); return SIGNED; }
__signed__          { set_text(); return SIGNED; }
__vector            { set_text(); return Ignore; }
__volatile__        { set_text(); return VOLATILE; }
__typeof            { set_text(); return TYPEOF; }
__typeof__          { set_text(); return TYPEOF; }
__int8              { set_text(); return INT8; }
__int16             { set_text(); return INT16; }
__int32             { set_text(); return INT32; }
__int64             { set_text(); return INT64; }
__ptr32             { set_text(); return PTR32; }
__ptr64             { set_text(); return PTR64; }
asm                 { set_text(); return ASM; }
auto                { set_text(); return AUTO; }
bool                { set_text(); return BOOLEAN; }
break               { set_text(); return BREAK; }
case                { set_text(); return CASE; }
catch               { set_text(); return CATCH; }
char                { set_text(); return CHAR; }
class               { set_text(); return CLASS; }
const               { set_text(); return CONST; }
continue            { set_text(); return CONTINUE; }
default             { set_text(); return DEFAULT; }
delete              { set_text(); return DELETE; }
do                  { set_text(); return DO; }
double              { set_text(); return DOUBLE; }
else                { set_text(); return ELSE; }
enum                { set_text(); return ENUM; }
explicit            { set_text(); return EXPLICIT; }
extern              { set_text(); return EXTERN; }
float               { set_text(); return FLOAT; }
for                 { set_text(); return FOR; }
friend              { set_text(); return FRIEND; }
goto                { set_text(); return GOTO; }
if                  { set_text(); return IF; }
inline              { set_text(); return INLINE; }
int                 { set_text(); return INT; }
long                { set_text(); return LONG; }
mutable             { set_text(); return MUTABLE; }
namespace           { set_text(); return NAMESPACE; }
new                 { set_text(); return NEW; }
noreturn            { set_text(); return Ignore; }
operator            { set_text(); return OPERATOR; }
private             { set_text(); return PRIVATE; }
protected           { set_text(); return PROTECTED; }
public              { set_text(); return PUBLIC; }
register            { set_text(); return REGISTER; }
return              { set_text(); return RETURN; }
short               { set_text(); return SHORT; }
signed              { set_text(); return SIGNED; }
sizeof              { set_text(); return SIZEOF; }
static              { set_text(); return STATIC; }
struct              { set_text(); return STRUCT; }
switch              { set_text(); return SWITCH; }
template            { set_text(); return TEMPLATE; }
this                { set_text(); return THIS; }
throw               { set_text(); return THROW; }
try                 { set_text(); return TRY; }
typedef             { set_text(); return TYPEDEF; }
typeid              { set_text(); return TYPEID; }
typename            { set_text(); return TYPENAME; }
union               { set_text(); return UNION; }
unsigned            { set_text(); return UNSIGNED; }
using               { set_text(); return USING; }
virtual             { set_text(); return VIRTUAL; }
void                { set_text(); return VOID; }
volatile            { set_text(); return VOLATILE; }
wchar_t	            { set_text(); return WCHAR_T; }
while               { set_text(); return WHILE; }

{identifier}        { set_text();
                      return Identifier; // or TYPDEFname
                    }

{decimal_constant}  { set_text();
                      convert_integer_literal(yytext, PARSER.current_token().data, 10);
                      return Constant;
                    }

{octal_constant}    { set_text();
                      convert_integer_literal(yytext, PARSER.current_token().data, 8);
                      return Constant;
                    }

{hex_constant}      { set_text();
                      convert_integer_literal(yytext, PARSER.current_token().data, 16);
                      return Constant;
                    }

{floating_constant} {
                      set_text();
                      convert_float_literal(yytext, PARSER.current_token().data);
                      return Constant;
                    }

"L"?[']{c_char}+['] { 
                      set_text();
                      convert_character_literal(yytext, PARSER.current_token().data);
                      return Constant;
                    }

"L"?["]{s_char}*["] {
                      set_text();
                      convert_string_literal(yytext, PARSER.current_token().data);
                      return StringL;
                    }

"("                 { set_text(); return yytext[0]; }
")"                 { set_text(); return yytext[0]; }
","                 { set_text(); return yytext[0]; }

{cpplineno}         { set_line_no(); }
{pragma}            { /* ignore it */ }

"#"

"#".*"\n"           { yycpperror("unknown preprocessor directive"); }

"##"                { }

"{"                 { set_text(); return '{'; }
"}"                 { set_text(); return '}'; }
"["                 { set_text(); return '['; }
"]"                 { set_text(); return ']'; }
"."                 { set_text(); return '.'; }
"&"                 { set_text(); return '&'; }
"*"                 { set_text(); return '*'; }
"+"                 { set_text(); return '+'; }
"-"                 { set_text(); return '-'; }
"~"                 { set_text(); return '~'; }
"!"                 { set_text(); return '!'; }
"/"                 { set_text(); return '/'; }
"%"                 { set_text(); return '%'; }
"<"                 { set_text(); return '<'; }
">"                 { set_text(); return '>'; }
"^"                 { set_text(); return '^'; }
"|"                 { set_text(); return '|'; }
"?"                 { set_text(); return '?'; }
":"                 { set_text(); return ':'; }
";"                 { set_text(); return ';'; }
"="                 { set_text(); return '='; }

".*"                { set_text(); return PmOp; }
"->*"               { set_text(); return PmOp; }
"::"                { set_text(); return Scope; }
"->"                { set_text(); return ArrowOp; }
"++"                { set_text(); return IncOp; }
"--"                { set_text(); return IncOp; }
"<<"                { set_text(); return ShiftOp; }
">>"                { set_text(); return ShiftOp; }
"<="                { set_text(); return RelOp; }
">="                { set_text(); return RelOp; }
"=="                { set_text(); return EqualOp; }
"!="                { set_text(); return EqualOp; }
"&&"                { set_text(); return LogAndOp; }
"||"                { set_text(); return LogOrOp; }
"*="                { set_text(); return AssignOp; }
"/="                { set_text(); return AssignOp; }
"%="                { set_text(); return AssignOp; }
"+="                { set_text(); return AssignOp; }
"-="                { set_text(); return AssignOp; }
"<<="               { set_text(); return AssignOp; }
">>="               { set_text(); return AssignOp; }
"&="                { set_text(); return AssignOp; }
"^="                { set_text(); return AssignOp; }
"|="                { set_text(); return AssignOp; }
"..."               { set_text(); return Ellipsis; }

.                   { yycpperror("unknown character"); }
}

%%

int yywrap() { return 1; }

void set_line_no()
{
  preprocessor_line(yytext, PARSER.line_no, PARSER.filename);
}
