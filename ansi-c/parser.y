%{

/*
 * This parser is specified based on:
 *
 * c5.y, a ANSI-C grammar written by James A. Roskind.
 * "Portions Copyright (c) 1989, 1990 James A. Roskind".
 * (http://www.idiom.com/free-compilers/,
 * ftp://ftp.infoseek.com/ftp/pub/c++grammar/,
 * ftp://ftp.sra.co.jp/.a/pub/cmd/c++grammar2.0.tar.gz)
 */

#define PARSER ansi_c_parser

#include "ansi_c_parser.h"

int yyansi_clex();
extern char *yyansi_ctext;

#include "parser_static.inc"

#include "y.tab.h"

/*** token declaration **************************************************/
%}

/*** ANSI-C keywords ***/

%token	TOK_AUTO      "auto"
%token  TOK_BOOL      "bool"
%token	TOK_BREAK     "break"
%token	TOK_CASE      "case"
%token	TOK_CHAR      "char"
%token	TOK_CONST     "const"
%token	TOK_CONTINUE  "continue"
%token	TOK_DEFAULT   "default"
%token	TOK_DO        "do"
%token	TOK_DOUBLE    "double"
%token	TOK_ELSE      "else"
%token	TOK_ENUM      "enum"
%token	TOK_EXTERN    "extern"
%token	TOK_FLOAT     "float"
%token	TOK_FOR       "for"
%token	TOK_GOTO      "goto"
%token	TOK_IF        "if"
%token  TOK_INLINE    "inline"
%token	TOK_INT       "int"
%token	TOK_LONG      "long"
%token	TOK_REGISTER  "register"
%token	TOK_RETURN    "return"
%token	TOK_SHORT     "short"
%token	TOK_SIGNED    "signed"
%token	TOK_SIZEOF    "sizeof"
%token	TOK_STATIC    "static"
%token	TOK_STRUCT    "struct"
%token	TOK_SWITCH    "switch"
%token	TOK_TYPEDEF   "typedef"
%token	TOK_UNION     "union"
%token	TOK_UNSIGNED  "unsigned"
%token	TOK_VOID      "void"
%token	TOK_VOLATILE  "volatile"
%token	TOK_WHILE     "while"

/*** multi-character operators ***/

%token	TOK_ARROW
%token	TOK_INCR
%token	TOK_DECR
%token	TOK_SHIFTLEFT
%token	TOK_SHIFTRIGHT
%token	TOK_LE
%token	TOK_GE
%token	TOK_EQ
%token	TOK_NE
%token	TOK_ANDAND
%token	TOK_OROR
%token	TOK_ELLIPSIS

/*** modifying assignment operators ***/

%token	TOK_MULTASSIGN
%token	TOK_DIVASSIGN
%token	TOK_MODASSIGN
%token	TOK_PLUSASSIGN
%token	TOK_MINUSASSIGN
%token	TOK_SLASSIGN
%token	TOK_SRASSIGN
%token	TOK_ANDASSIGN
%token	TOK_EORASSIGN
%token	TOK_ORASSIGN

/*** scanner parsed tokens (these have a value!) ***/

%token	TOK_IDENTIFIER
%token	TOK_TYPEDEFNAME
%token	TOK_INTEGER
%token	TOK_FLOATING
%token	TOK_CHARACTER
%token	TOK_STRING

/*** extensions ***/

%token	TOK_INT8
%token	TOK_INT16
%token	TOK_INT32
%token	TOK_INT64
%token	TOK_PTR32
%token	TOK_PTR64
%token  TOK_TYPEOF
%token  TOK_GCC_ASM
%token  TOK_MSC_ASM
%token	TOK_BUILTIN_VA_ARG
%token	TOK_BUILTIN_OFFSETOF

/*** special scanner reports ***/

%token	TOK_SCANNER_ERROR	/* used by scanner to report errors */
%token	TOK_SCANNER_EOF		/* used by scanner to report end of import */

/*** grammar selection ***/

%token	TOK_PARSE_LANGUAGE
%token	TOK_PARSE_EXPRESSION
%token	TOK_PARSE_TYPE

/*** priority, associativity, etc. definitions **************************/


%start	grammar

%expect 10	/* the famous "dangling `else'" ambiguity */
		/* results in one shift/reduce conflict   */
		/* that we don't want to be reported      */
		/* PLUS +2: KnR ambiguity */

%{
/************************************************************************/
/*** rules **************************************************************/
/************************************************************************/
%}
%%

/*** Grammar selection **************************************************/

grammar: TOK_PARSE_LANGUAGE translation_unit
	| TOK_PARSE_EXPRESSION comma_expression
	{
	  PARSER.parse_tree.declarations.push_back(ansi_c_declarationt());
	  PARSER.parse_tree.declarations.back().swap(stack($2));
	}
	| TOK_PARSE_TYPE type_name
	;

/*** Token with values **************************************************/


identifier:
	TOK_IDENTIFIER
	;

typedef_name:
	TOK_TYPEDEFNAME
	;

integer:
	TOK_INTEGER
	;

floating:
	TOK_FLOATING
	;

character:
	TOK_CHARACTER
	;

string:
	TOK_STRING
	;

/*** Constants **********************************************************/

/* note: the following has been changed from the ANSI-C grammar:	*/
/*	- constant includes string_literal_list (cleaner)		*/

constant:
	integer
	| floating
	| character
	| string_literal_list
	;

string_literal_list:
	string
	| string_literal_list string
	{ $$ = $1;
	  // do concatenation
	  stack($$).set("value", stack($$).get_string("value")+
	    stack($2).get_string("value"));
	}
	;

/*** Expressions ********************************************************/

primary_expression:
	identifier
	| constant
	| '(' comma_expression ')'
	{ $$ = $2; }
	| statement_expression
	| builtin_va_arg_expression
	| builtin_offsetof
	;

builtin_va_arg_expression:
	TOK_BUILTIN_VA_ARG '(' assignment_expression ',' type_name ')'
	{
	  $$=$1;
	  stack($$).id("builtin_va_arg");
	  mto($$, $3);
	  stack($$).type().swap(stack($5));
	}
	;

builtin_offsetof:
	TOK_BUILTIN_OFFSETOF '(' type_name ',' offsetof_member_designator ')'
	{
	  $$=$1;
	  stack($$).id("builtin_offsetof");
	  stack($$).add("offsetof_type").swap(stack($3));
	  stack($$).add("member").swap(stack($5));
	}
	;

offsetof_member_designator:
          member_name
        | offsetof_member_designator '.' member_name
        | offsetof_member_designator '[' comma_expression ']'
        ;                  

statement_expression: '(' compound_statement ')'
	{ init($$, "sideeffect");
	  stack($$).set("statement", "statement_expression");
          mto($$, $2);
	}
	;

postfix_expression:
	primary_expression
	| postfix_expression '[' comma_expression ']'
	{ binary($$, $1, $2, "index", $3); }
	| postfix_expression '(' ')'
	{ $$=$2;
	  set($$, "sideeffect");
	  stack($$).operands().resize(2);
	  stack($$).op0().swap(stack($1));
	  stack($$).op1().clear();
	  stack($$).op1().id("arguments");
	  stack($$).set("statement", "function_call");
	}
	| postfix_expression '(' argument_expression_list ')'
	{ $$=$2;
	  init($$, "sideeffect");
	  stack($$).set("statement", "function_call");
	  stack($$).operands().resize(2);
	  stack($$).op0().swap(stack($1));
	  stack($$).op1().swap(stack($3));
	  stack($$).op1().id("arguments");
	}
	| postfix_expression '.' member_name
	{ $$=$2;
	  set($$, "member");
	  mto($$, $1);
	  stack($$).set("component_name", stack($3).get("#base_name"));
	}
	| postfix_expression TOK_ARROW member_name
	{ $$=$2;
	  set($$, "ptrmember");
	  mto($$, $1);
	  stack($$).set("component_name", stack($3).get("#base_name"));
	}
	| postfix_expression TOK_INCR
	{ $$=$2;
	  init($$, "sideeffect");
	  mto($$, $1);
	  stack($$).set("statement", "postincrement");
	}
	| postfix_expression TOK_DECR
	{ $$=$2;
	  init($$, "sideeffect");
	  mto($$, $1);
	  stack($$).set("statement", "postdecrement");
	}
	;

member_name:
	identifier
	| typedef_name
	;

argument_expression_list:
	assignment_expression
	{
	  init($$, "expression_list");
	  mto($$, $1);
	}
	| argument_expression_list ',' assignment_expression
	{
	  $$=$1;
	  mto($$, $3);
	}
	;

unary_expression:
	postfix_expression
	| TOK_INCR unary_expression
	{ $$=$1;
	  set($$, "sideeffect");
	  stack($$).set("statement", "preincrement");
	  mto($$, $2);
	}
	| TOK_DECR unary_expression
	{ $$=$1;
	  set($$, "sideeffect");
	  stack($$).set("statement", "predecrement");
	  mto($$, $2);
	}
	| '&' cast_expression
	{ $$=$1;
	  set($$, "address_of");
	  mto($$, $2);
	}
	| '*' cast_expression
	{ $$=$1;
	  set($$, "dereference");
	  mto($$, $2);
	}
	| '+' cast_expression
	{ $$=$1;
	  set($$, "unary+");
	  mto($$, $2);
	}
	| '-' cast_expression
	{ $$=$1;
	  set($$, "unary-");
	  mto($$, $2);
	}
	| '~' cast_expression
	{ $$=$1;
	  set($$, "bitnot");
	  mto($$, $2);
	}
	| '!' cast_expression
	{ $$=$1;
	  set($$, "not");
	  mto($$, $2);
	}
	| TOK_SIZEOF unary_expression
	{ $$=$1;
	  set($$, "sizeof");
	  mto($$, $2);
	}
	| TOK_SIZEOF '(' type_name ')'
	{ $$=$1;
	  set($$, "sizeof");
	  stack($$).add("sizeof-type").swap(stack($3));
	}
	;

cast_expression:
	unary_expression
	| '(' type_name ')' cast_expression
	{
	  $$=$1;
	  set($$, "typecast");
	  mto($$, $4);
	  stack($$).type().swap(stack($2));
	}
	/* The following is a GCC extension
	   to allow a 'temporary union' */
	| '(' type_name ')' '{' designated_initializer_list '}'
	{
	  exprt tmp("designated_list");
	  tmp.operands().swap(stack($5).operands());
	  $$=$1;
	  set($$, "typecast");
	  stack($$).move_to_operands(tmp);
	  stack($$).type().swap(stack($2));
	}
	;

multiplicative_expression:
	cast_expression
	| multiplicative_expression '*' cast_expression
	{ binary($$, $1, $2, "*", $3); }
	| multiplicative_expression '/' cast_expression
	{ binary($$, $1, $2, "/", $3); }
	| multiplicative_expression '%' cast_expression
	{ binary($$, $1, $2, "mod", $3); }
	;

additive_expression:
	multiplicative_expression
	| additive_expression '+' multiplicative_expression
	{ binary($$, $1, $2, "+", $3); }
	| additive_expression '-' multiplicative_expression
	{ binary($$, $1, $2, "-", $3); }
	;

shift_expression:
	additive_expression
	| shift_expression TOK_SHIFTLEFT additive_expression
	{ binary($$, $1, $2, "shl", $3); }
	| shift_expression TOK_SHIFTRIGHT additive_expression
	{ binary($$, $1, $2, "shr", $3); }
	;

relational_expression:
	shift_expression
	| relational_expression '<' shift_expression
	{ binary($$, $1, $2, "<", $3); }
	| relational_expression '>' shift_expression
	{ binary($$, $1, $2, ">", $3); }
	| relational_expression TOK_LE shift_expression
	{ binary($$, $1, $2, "<=", $3); }
	| relational_expression TOK_GE shift_expression
	{ binary($$, $1, $2, ">=", $3); }
	;

equality_expression:
	relational_expression
	| equality_expression TOK_EQ relational_expression
	{ binary($$, $1, $2, "=", $3); }
	| equality_expression TOK_NE relational_expression
	{ binary($$, $1, $2, "notequal", $3); }
	;

and_expression:
	equality_expression
	| and_expression '&' equality_expression
	{ binary($$, $1, $2, "bitand", $3); }
	;

exclusive_or_expression:
	and_expression
	| exclusive_or_expression '^' and_expression
	{ binary($$, $1, $2, "bitxor", $3); }
	;

inclusive_or_expression:
	exclusive_or_expression
	| inclusive_or_expression '|' exclusive_or_expression
	{ binary($$, $1, $2, "bitor", $3); }
	;

logical_and_expression:
	inclusive_or_expression
	| logical_and_expression TOK_ANDAND inclusive_or_expression
	{ binary($$, $1, $2, "and", $3); }
	;

logical_or_expression:
	logical_and_expression
	| logical_or_expression TOK_OROR logical_and_expression
	{ binary($$, $1, $2, "or", $3); }
	;

conditional_expression:
	logical_or_expression
	| logical_or_expression '?' comma_expression ':' conditional_expression
	{ $$=$2;
	  init($$, "if");
	  mto($$, $1);
	  mto($$, $3);
	  mto($$, $5);
	}
	| logical_or_expression '?' ':' conditional_expression
	{ $$=$2;
	  init($$, "sideeffect");
	  stack($$).set("statement", "gcc_conditional_expression");
	  mto($$, $1);
	  mto($$, $4);
	}
	;

assignment_expression:
	conditional_expression
	| cast_expression '=' assignment_expression
	{ binary($$, $1, $2, "sideeffect", $3); stack($$).set("statement", "assign"); }
	| cast_expression TOK_MULTASSIGN assignment_expression
	{ binary($$, $1, $2, "sideeffect", $3); stack($$).set("statement", "assign*"); }
	| cast_expression TOK_DIVASSIGN assignment_expression
	{ binary($$, $1, $2, "sideeffect", $3); stack($$).set("statement", "assign_div"); }
	| cast_expression TOK_MODASSIGN assignment_expression
	{ binary($$, $1, $2, "sideeffect", $3); stack($$).set("statement", "assign_mod"); }
	| cast_expression TOK_PLUSASSIGN assignment_expression
	{ binary($$, $1, $2, "sideeffect", $3); stack($$).set("statement", "assign+"); }
	| cast_expression TOK_MINUSASSIGN assignment_expression
	{ binary($$, $1, $2, "sideeffect", $3); stack($$).set("statement", "assign-"); }
	| cast_expression TOK_SLASSIGN assignment_expression
	{ binary($$, $1, $2, "sideeffect", $3); stack($$).set("statement", "assign_shl"); }
	| cast_expression TOK_SRASSIGN assignment_expression
	{ binary($$, $1, $2, "sideeffect", $3); stack($$).set("statement", "assign_shr"); }
	| cast_expression TOK_ANDASSIGN assignment_expression
	{ binary($$, $1, $2, "sideeffect", $3); stack($$).set("statement", "assign_bitand"); }
	| cast_expression TOK_EORASSIGN assignment_expression
	{ binary($$, $1, $2, "sideeffect", $3); stack($$).set("statement", "assign_bitxor"); }
	| cast_expression TOK_ORASSIGN assignment_expression
	{ binary($$, $1, $2, "sideeffect", $3); stack($$).set("statement", "assign_bitor"); }
	;

comma_expression:
	assignment_expression
	| comma_expression ',' assignment_expression
	{ binary($$, $1, $2, "comma", $3); }
	;

constant_expression:
	assignment_expression
	;

comma_expression_opt:
	/* nothing */
	{ init($$); stack($$).make_nil(); }
	| comma_expression
	;

/*** Declarations *******************************************************/


declaration:
	declaration_specifier ';'
	{
	  init($$);
	}
	| type_specifier ';'
	{
	  init($$);
	}
	| declaring_list ';'
	| default_declaring_list ';'
	;

default_declaring_list:
	declaration_qualifier_list identifier_declarator
		{
		  init($$);
		  PARSER.new_declaration(stack($1), stack($2), stack($$));
		}
	initializer_opt
		{
		  init($$);
		  stack($$).add("type")=stack($1);
		  decl_statement($$, $3, $4);
		}
	| type_qualifier_list identifier_declarator
	{
	  init($$);
	  PARSER.new_declaration(stack($1), stack($2), stack($$));
	}
	initializer_opt
	{
	  init($$);
	  stack($$).add("type")=stack($1);
	  decl_statement($$, $3, $4);
	}
	| default_declaring_list ',' identifier_declarator
		{
		  init($$);
		  const irept &t=stack($1).find("type");
		  PARSER.new_declaration(t, stack($3), stack($$));
		}
		initializer_opt
	{
	  $$=$1;
	  decl_statement($$, $4, $5);
	}
	;

declaring_list:			/* DeclarationSpec */
	declaration_specifier declarator
		{
		  // the symbol has to be visible during initialization
		  init($$);
		  PARSER.new_declaration(stack($1), stack($2), stack($$));
		}
		initializer_opt
	{
	  init($$);
	  stack($$).add("type")=stack($1);
	  decl_statement($$, $3, $4);
	}
	| type_specifier declarator
		{
		  // the symbol has to be visible during initialization
		  init($$);
		  PARSER.new_declaration(stack($1), stack($2), stack($$));
		}
		initializer_opt
	{
	  init($$);
	  stack($$).add("type")=stack($1);
	  decl_statement($$, $3, $4);
	}
	| declaring_list ',' declarator
		{
		  init($$);
		  const irept &t=stack($1).find("type");
		  PARSER.new_declaration(t, stack($3), stack($$));
		}
		initializer_opt
	{
	  $$=$1;
	  decl_statement($$, $4, $5);
	}
	;

declaration_specifier:
	basic_declaration_specifier
	| sue_declaration_specifier
	| typedef_declaration_specifier
	;

type_specifier:
	basic_type_specifier
	| sue_type_specifier
	| typedef_type_specifier
	| typeof_type_specifier
	;

declaration_qualifier_list:
	storage_class
	| type_qualifier_list storage_class
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	| declaration_qualifier_list declaration_qualifier
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	;

type_qualifier_list:
	type_qualifier
	| type_qualifier_list type_qualifier
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	;

declaration_qualifier:
	storage_class
	| type_qualifier
	;

type_qualifier:
	TOK_CONST      { $$=$1; set($$, "const"); }
	| TOK_VOLATILE { $$=$1; set($$, "volatile"); }
	;

basic_declaration_specifier:
	declaration_qualifier_list basic_type_name
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	| basic_type_specifier storage_class
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	| basic_declaration_specifier declaration_qualifier
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	| basic_declaration_specifier basic_type_name
	{
	  $$=$1;
	  merge_types($$, $2);
	};

basic_type_specifier:
	basic_type_name
	| type_qualifier_list basic_type_name
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	| basic_type_specifier type_qualifier
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	| basic_type_specifier basic_type_name
	{
	  $$=$1;
	  merge_types($$, $2);
	};

sue_declaration_specifier:
	declaration_qualifier_list elaborated_type_name
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	| sue_type_specifier storage_class
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	| sue_declaration_specifier declaration_qualifier
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	;

sue_type_specifier:
	elaborated_type_name
	| type_qualifier_list elaborated_type_name
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	| sue_type_specifier type_qualifier
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	;

typedef_declaration_specifier:	/* DeclarationSpec */
	typedef_type_specifier storage_class
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	| declaration_qualifier_list typedef_name
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	| typedef_declaration_specifier declaration_qualifier
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	;

typedef_type_specifier:		/* Type */
	typedef_name
	| type_qualifier_list typedef_name
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	| typedef_type_specifier type_qualifier
	{
	  $$=$1;
	  merge_types($$, $2);
	}
	;

typeof_type_specifier:
	TOK_TYPEOF '(' comma_expression ')'
	{ $$ = $3;
	  locationt location=stack($$).location();
	  typet new_type("type_of");
	  new_type.subtype() = (typet &)(stack($$));
	  stack($$).swap(new_type);
	  stack($$).location()=location;
	  stack($$).set("#is_expression", true);
	}
	| TOK_TYPEOF '(' ptr_type_specifier  ')'
	{ $$ = $3;
	  locationt location=stack($$).location();
	  typet new_type("type_of");
	  new_type.subtype() = (typet &)(stack($$));
	  stack($$).swap(new_type);
	  stack($$).location()=location;
	  stack($$).set("#is_expression", false);
	}
	;

ptr_type_specifier:
	type_specifier
	| ptr_type_specifier '*'
	{ $$ = $1;
	  locationt location=stack($$).location();
	  typet new_type("pointer");
	  new_type.subtype() = (typet&) stack($$);
	  stack($$).swap(new_type);
	  stack($$).location()=location;
	}
	;

storage_class:
	TOK_TYPEDEF    { $$=$1; set($$, "typedef"); }
	| TOK_EXTERN   { $$=$1; set($$, "extern"); }
	| TOK_STATIC   { $$=$1; set($$, "static"); }
	| TOK_AUTO     { $$=$1; set($$, "auto"); }
	| TOK_REGISTER { $$=$1; set($$, "register"); }
	| TOK_INLINE   { $$=$1; set($$, "inline"); }
	;

basic_type_name:
	TOK_INT        { $$=$1; set($$, "int"); }
	| TOK_INT8     { $$=$1; set($$, "int8"); }
	| TOK_INT16    { $$=$1; set($$, "int16"); }
	| TOK_INT32    { $$=$1; set($$, "int32"); }
	| TOK_INT64    { $$=$1; set($$, "int64"); }
	| TOK_PTR32    { $$=$1; set($$, "ptr32"); }
	| TOK_PTR64    { $$=$1; set($$, "ptr64"); }
	| TOK_CHAR     { $$=$1; set($$, "char"); }
	| TOK_SHORT    { $$=$1; set($$, "short"); }
	| TOK_LONG     { $$=$1; set($$, "long"); }
	| TOK_FLOAT    { $$=$1; set($$, "float"); }
	| TOK_DOUBLE   { $$=$1; set($$, "double"); }
	| TOK_SIGNED   { $$=$1; set($$, "signed"); }
	| TOK_UNSIGNED { $$=$1; set($$, "unsigned"); }
	| TOK_VOID     { $$=$1; set($$, "empty"); }
	| TOK_BOOL     { $$=$1; set($$, "bool"); }
	;

elaborated_type_name:
	aggregate_name
	| enum_name
	;

aggregate_name:
	aggregate_key
		{
		  // an anon struct
		  exprt symbol("symbol");

		  symbol.set("#base_name", PARSER.get_anon_name());

		  init($$);
		  PARSER.new_declaration(stack($1), symbol, stack($$), true);
		}
		'{' member_declaration_list_opt '}'
	{
	  typet &type=stack($2).type();
	  type.add("components").get_sub().swap(stack($4).add("operands").get_sub());

	  // grab symbol
	  init($$, "symbol");
	  stack($$).set("identifier", stack($2).get("name"));
	  stack($$).location()=stack($2).location();

	  PARSER.move_declaration(stack($2));
	}
	| aggregate_key identifier_or_typedef_name
		{
		  PARSER.new_declaration(stack($1), stack($2), stack($$), true);

		  exprt tmp(stack($$));
		  tmp.type().id("incomplete_"+tmp.type().id_string());
		  PARSER.move_declaration(tmp);
		}
		'{' member_declaration_list_opt '}'
	{
	  typet &type=stack($3).type();
	  type.add("components").get_sub().swap(stack($5).add("operands").get_sub());

	  // grab symbol
	  init($$, "symbol");
	  stack($$).set("identifier", stack($3).get("name"));
	  stack($$).location()=stack($3).location();

	  PARSER.move_declaration(stack($3));
	}
	| aggregate_key identifier_or_typedef_name
	{
	  do_tag($1, $2);
	  $$=$2;
	}
	;

aggregate_key:
	TOK_STRUCT
	{ $$=$1; set($$, "struct"); }
	| TOK_UNION
	{ $$=$1; set($$, "union"); }
	;

member_declaration_list_opt:
		  /* Nothing */
	{
	  init($$, "declaration_list");
	}
	| member_declaration_list
	;

member_declaration_list:
	  member_declaration
	| member_declaration_list member_declaration
	{
	  assert(stack($1).id()=="declaration_list");
	  assert(stack($2).id()=="declaration_list");
	  $$=$1;
	  Forall_operands(it, stack($2))
	    stack($$).move_to_operands(*it);
	  stack($2).clear();
	}
	;

member_declaration:
	member_declaring_list ';'
	| member_default_declaring_list ';'
	| ';' /* empty declaration */
	{
	  init($$, "declaration_list");
	}
	;

member_default_declaring_list:
	type_qualifier_list member_identifier_declarator
	{
	  init($$, "declaration_list");

	  exprt declaration;

	  PARSER.new_declaration(stack($1), stack($2), declaration, false, false);

	  stack($$).move_to_operands(declaration);
	}
	| member_default_declaring_list ',' member_identifier_declarator
	{
	  exprt declaration;

	  typet type;
	  PARSER.new_declaration(stack($1), stack($3), declaration, false, false);

	  $$=$1;
	  stack($$).move_to_operands(declaration);
	}
	;

member_declaring_list:
	type_specifier member_declarator
	{
	  init($$, "declaration_list");

	  // save the type_specifier
	  stack($$).add("declaration_type")=stack($1);

	  exprt declaration;
	  PARSER.new_declaration(stack($1), stack($2), declaration, false, false);

	  stack($$).move_to_operands(declaration);
	}
	| member_declaring_list ',' member_declarator
	{
	  exprt declaration;

	  irept declaration_type(stack($1).find("declaration_type"));
	  PARSER.new_declaration(declaration_type, stack($3), declaration, false, false);

	  $$=$1;
	  stack($$).move_to_operands(declaration);
	}
	;

member_declarator:
	declarator bit_field_size_opt
	{
	  if(!stack($2).is_nil())
	  {
	    $$=$2;
	    stack($$).add("subtype").swap(stack($1));
	  }
	  else
	    $$=$1;
	}
	| /* empty */
	{
	  init($$);
	  stack($$).make_nil();
	}
	| bit_field_size
	{
	  $$=$1;
	  stack($$).add("subtype").make_nil();
	}
	;

member_identifier_declarator:
	identifier_declarator
		{ /* note: this mid-rule action (suggested by the grammar) */
		  /*       is not used because we don't have direct access */
		  /*       to the declaration specifier; therefore the     */
		  /*       symbol table is not updated ASAP (which is only */
		  /*       a minor problem; bit_field_size_opt expression  */
		  /*       cannot use the identifier_declarator)           */
		 }
		bit_field_size_opt
	{
	  $$=$1;
	  if(!stack($3).is_nil())
	    merge_types($$, $3);
	}
	| bit_field_size
	{
	  // TODO
	  assert(0);
	}
	;

bit_field_size_opt:
	/* nothing */
	{
	  init($$);
	  stack($$).make_nil();
	}
	| bit_field_size
	;

bit_field_size:			/* Expression */
	':' constant_expression
	{
	  $$=$1; set($$, "c_bitfield");
	  stack($$).set("size", stack($2));
	}
	;

/* note: although the grammar didn't suggest mid-rule actions here	*/
/*       we handle enum exactly like struct/union			*/
enum_name:			/* Type */
	enum_key
		{
		  // an anon enum
		  exprt symbol("symbol");
		  symbol.set("#base_name", PARSER.get_anon_name());

		  PARSER.new_declaration(stack($1), symbol, stack($$), true);

		  exprt tmp(stack($$));
		  PARSER.move_declaration(tmp);
		}
		'{' enumerator_list '}'
	{
	  // grab symbol
	  init($$, "symbol");
	  stack($$).set("identifier", stack($2).get("name"));
	  stack($$).location()=stack($2).location();

	  do_enum_members((const typet &)stack($$), stack($4));

	  PARSER.move_declaration(stack($2));
	}
	| enum_key identifier_or_typedef_name
		{ /* !!! mid-rule action !!! */
		  PARSER.new_declaration(stack($1), stack($2), stack($$), true);

		  exprt tmp(stack($$));
		  PARSER.move_declaration(tmp);
		}
		'{' enumerator_list '}'
	{
	  // grab symbol
	  init($$, "symbol");
	  stack($$).set("identifier", stack($3).get("name"));
	  stack($$).location()=stack($3).location();

	  do_enum_members((const typet &)stack($$), stack($5));

	  PARSER.move_declaration(stack($3));
	}
	| enum_key identifier_or_typedef_name
	{
	  do_tag($1, $2);
	  $$=$2;
	}
	;

enum_key: TOK_ENUM
	{
	  $$=$1;
	  set($$, "c_enum");
	}
	;

enumerator_list:		/* MemberList */
	enumerator_declaration
	{
	  init($$);
	  mto($$, $1);
	}
	| enumerator_list ',' enumerator_declaration
	{
	  $$=$1;
	  mto($$, $3);
	}
	| enumerator_list ','
	{
	  $$=$1;
	}
	;

enumerator_declaration:
	  identifier_or_typedef_name enumerator_value_opt
	{
	  init($$);
	  irept type("enum");
	  PARSER.new_declaration(type, stack($1), stack($$));
	  stack($$).set("is_macro", true);
	  stack($$).add("value").swap(stack($2));
	}
	;

enumerator_value_opt:		/* Expression */
	/* nothing */
	{
	  init($$);
	  stack($$).make_nil();
	}
	| '=' constant_expression
	{
	  $$=$2;
	}
	;

parameter_type_list:		/* ParameterList */
	parameter_list
	| parameter_list ',' TOK_ELLIPSIS
	{
	  typet tmp("ansi_c_ellipsis");
	  $$=$1;
	  ((typet &)stack($$)).move_to_subtypes(tmp);
	}
	| KnR_parameter_list
	;

KnR_parameter_list:
	KnR_parameter
	{
          init($$, "arguments");
          mts($$, $1);
	}
	| KnR_parameter_list ',' KnR_parameter
	{
          $$=$1;
          mts($$, $3);
	}
	;

KnR_parameter: identifier
	{
          init($$);
	  irept type("KnR");
	  PARSER.new_declaration(type, stack($1), stack($$));
	}
	;

parameter_list:
	parameter_declaration
	{
	  init($$, "arguments");
	  mts($$, $1);
	}
	| parameter_list ',' parameter_declaration
	{
	  $$=$1;
	  mts($$, $3);
	}
	;

parameter_declaration:
	declaration_specifier
	{
	  init($$);
	  exprt nil;
	  nil.make_nil();
	  PARSER.new_declaration(stack($1), nil, stack($$));
	}
	| declaration_specifier parameter_abstract_declarator
	{
	  init($$);
	  PARSER.new_declaration(stack($1), stack($2), stack($$));
	}
	| declaration_specifier identifier_declarator
	{
	  init($$);
	  PARSER.new_declaration(stack($1), stack($2), stack($$));
	}
	| declaration_specifier parameter_typedef_declarator
	{
          // the second tree is really the argument -- not part
          // of the type!
	  init($$);
	  PARSER.new_declaration(stack($1), stack($2), stack($$));
	}
	| declaration_qualifier_list
	{
	  init($$);
	  exprt nil;
	  nil.make_nil();
	  PARSER.new_declaration(stack($1), nil, stack($$));
	}
	| declaration_qualifier_list parameter_abstract_declarator
	{
	  init($$);
	  PARSER.new_declaration(stack($1), stack($2), stack($$));
	}
	| declaration_qualifier_list identifier_declarator
	{
	  init($$);
	  PARSER.new_declaration(stack($1), stack($2), stack($$));
	}
	| type_specifier
	{
	  init($$);
	  exprt nil;
	  nil.make_nil();
	  PARSER.new_declaration(stack($1), nil, stack($$));
	}
	| type_specifier parameter_abstract_declarator
	{
	  init($$);
	  PARSER.new_declaration(stack($1), stack($2), stack($$));
	}
	| type_specifier identifier_declarator
	{
	  init($$);
	  PARSER.new_declaration(stack($1), stack($2), stack($$));
	}
	| type_specifier parameter_typedef_declarator
	{
          // the second tree is really the argument -- not part
          // of the type!
	  init($$);
	  PARSER.new_declaration(stack($1), stack($2), stack($$));
	}
	| type_qualifier_list
	{
	  init($$);
	  exprt nil;
	  nil.make_nil();
	  PARSER.new_declaration(stack($1), nil, stack($$));
	}
	| type_qualifier_list parameter_abstract_declarator
	{
	  init($$);
	  PARSER.new_declaration(stack($1), stack($2), stack($$));
	}
	| type_qualifier_list identifier_declarator
	{
	  init($$);
	  PARSER.new_declaration(stack($1), stack($2), stack($$));
	}
	;

identifier_or_typedef_name:
	identifier
	| typedef_name
	;

type_name:
	type_specifier
	| type_specifier abstract_declarator
	{
	  $$=$1;
	  make_subtype($$, $2);
	}
	| type_qualifier_list
	| type_qualifier_list abstract_declarator
	{
	  $$=$1;
	  make_subtype($$, $2);
	}
	;

initializer_opt:
	/* nothing */
	{
	  newstack($$);
	  stack($$).make_nil();
	}
	| '=' initializer
	{ $$ = $2; }
	;

/* note: the following has been changed from the ANSI-C grammar:	*/
/*	- an initializer is not an assignment_expression,		*/
/*	  but a constant_expression					*/
/*	  (which probably is the case anyway for 99.9% of C programs)	*/

initializer:
	'{' initializer_list '}'
	{
	  $$=$1;
	  set($$, "constant");
	  stack($$).type().id("incomplete_array");
	  stack($$).operands().swap(stack($2).operands());
	}
	| '{' initializer_list ',' '}'
	{
	  $$=$1;
	  set($$, "constant");
	  stack($$).type().id("incomplete_array");
	  stack($$).operands().swap(stack($2).operands());
	}
	| constant_expression	/* was: assignment_expression */
	| '{' designated_initializer_list '}'
	{
	  $$=$1;
	  set($$, "designated_list");
	  stack($$).operands().swap(stack($2).operands());
	}
	;

initializer_list:
	initializer
	{
	  $$=$1;
	  exprt tmp;
	  tmp.swap(stack($$));
	  stack($$).clear();
	  stack($$).move_to_operands(tmp);
	}
	| initializer_list ',' initializer
	{
	  $$=$1;
	  mto($$, $3);
	}
	;

/* GCC extension: designated initializer */
designated_initializer:
          /* empty */
        {
	  newstack($$);
	  stack($$).make_nil();          
        }
        | '.' identifier '=' initializer
        {
          $$=$1;
          stack($$).id("designated_initializer");
          stack($$).set("component_name", stack($2).get("#base_name"));
          stack($$).move_to_operands(stack($4));
        }
        ;

designated_initializer_list:
	designated_initializer
	{
	  $$=$1;
	  exprt tmp;
	  tmp.swap(stack($$));
	  stack($$).clear();

	  if(tmp.is_not_nil())
            stack($$).move_to_operands(tmp);
	}
	| designated_initializer_list ',' designated_initializer
	{
	  $$=$1;
	  if(stack($3).is_not_nil())
	    mto($$, $3);
	}
	;

/*** Statements *********************************************************/

statement:
	  labeled_statement
	| compound_statement
	| declaration_statement
	| expression_statement
	| selection_statement
	| iteration_statement
	| jump_statement
	| gcc_asm_statement
	| msc_asm_statement
	;

declaration_statement:
	declaration
	{
	  init($$);
	  statement($$, "decl-block");
	  stack($$).operands().swap(stack($1).operands());
	}
	;

labeled_statement:
	identifier_or_typedef_name ':' statement
	{
	  $$=$2;
	  statement($$, "label");
	  stack($$).set("label", stack($1).get("#base_name"));
	  mto($$, $3);
	}
	| TOK_CASE constant_expression ':' statement
	{
	  $$=$1;
	  statement($$, "label");
	  mto($$, $4);
	  static_cast<exprt &>(stack($$).add("case")).
		move_to_operands(stack($2));
	}
	| TOK_DEFAULT ':' statement
	{
	  $$=$1;
	  statement($$, "label");
	  mto($$, $3);
	  stack($$).set("default", true);
	}
	;

/* note: the following has been changed from the ANSI-C grammar:	*/
/*	- rule compound_scope is used to prepare an inner scope for	*/
/*	  each compound_statement (and to obtain the line infos)	*/

compound_statement:
	compound_scope '{' '}'
	{
	  $$=$2;
	  statement($$, "block");
	  stack($$).set("#end_location", stack($3).location());
	  PARSER.pop_scope();
	}
	| compound_scope '{' statement_list '}'
	{
	  $$=$3;
	  stack($$).location()=stack($2).location();
	  stack($$).set("#end_location", stack($4).location());
	  PARSER.pop_scope();
	}
	;

compound_scope:
	/* nothing */
	{
	  unsigned prefix=++PARSER.current_scope().compound_counter;
	  PARSER.new_scope(i2string(prefix)+"::");
	}
	;

statement_list:
	statement
	{
	  $$=$1;
	  to_code(stack($$)).make_block();
	}
	| statement_list statement
	{
	  mto($$, $2);
	}
	;

expression_statement:
	comma_expression_opt ';'
	{
	  $$=$2;

	  if(stack($1).is_nil())
	    statement($$, "skip");
	  else
	  {
	    statement($$, "expression");
	    mto($$, $1);
	  }
	}
	;

selection_statement:
	  TOK_IF '(' comma_expression ')' statement
	{
	  $$=$1;
	  statement($$, "ifthenelse");
	  mto($$, $3);
	  mto($$, $5);
	}
	| TOK_IF '(' comma_expression ')' statement TOK_ELSE statement
	{
	  $$=$1;
	  statement($$, "ifthenelse");
	  mto($$, $3);
	  mto($$, $5);
	  mto($$, $7);
	}
	| TOK_SWITCH '(' comma_expression ')' statement
	{
	  $$=$1;
	  statement($$, "switch");
	  mto($$, $3);
	  mto($$, $5);
	}
	;

declaration_or_expression_statement:
	  declaration_statement
	| expression_statement
	;

iteration_statement:
	TOK_WHILE '(' comma_expression_opt ')' statement
	{
	  $$=$1;
	  statement($$, "while");
	  mto($$, $3);
	  mto($$, $5);
	}
	| TOK_DO statement TOK_WHILE '(' comma_expression ')' ';'
	{
	  $$=$1;
	  statement($$, "dowhile");
	  mto($$, $5);
	  mto($$, $2);
	}
	| TOK_FOR '(' declaration_or_expression_statement
		comma_expression_opt ';' comma_expression_opt ')' statement
	{
	  $$=$1;
	  statement($$, "for");
	  mto($$, $3);
	  mto($$, $4);
	  mto($$, $6);
	  mto($$, $8);
	}
	;

jump_statement:
	TOK_GOTO identifier_or_typedef_name ';'
	{
	  $$=$1;
	  statement($$, "goto");
	  stack($$).set("destination", stack($2).get("#base_name"));
	}
	| TOK_CONTINUE ';'
	{ $$=$1; statement($$, "continue"); }
	| TOK_BREAK ';'
	{ $$=$1; statement($$, "break"); }
	| TOK_RETURN ';'
	{ $$=$1; statement($$, "return"); }
	| TOK_RETURN comma_expression ';'
	{ $$=$1; statement($$, "return"); mto($$, $2); }
	;

gcc_asm_statement:
	TOK_GCC_ASM volatile_opt '(' asm_commands ')' ';'
	{ $$=$1;
	  statement($$, "asm");
	  stack($$).set("flavor", "gcc"); }
	;

msc_asm_statement:
	TOK_MSC_ASM '{' TOK_STRING '}'
	{ $$=$1;
	  statement($$, "asm"); 
	  stack($$).set("flavor", "msc"); }
	| TOK_MSC_ASM TOK_STRING
	{ $$=$1;
	  statement($$, "asm"); 
	  stack($$).set("flavor", "msc"); }
	;

volatile_opt:
          /* nothing */
        | TOK_VOLATILE
        ;

/* asm ( assembler template
           : output operands                  // optional
           : input operands                   // optional
           : list of clobbered registers      // optional
           );
*/

asm_commands:
          asm_assembler_template
        | asm_assembler_template asm_outputs
        | asm_assembler_template asm_outputs asm_inputs
        | asm_assembler_template asm_outputs asm_inputs asm_clobbered_registers
	;

asm_assembler_template: string_literal_list
        ;

asm_outputs:
          ':' asm_output_list
        ;

asm_output:
          string '(' comma_expression ')'
        ;

asm_output_list:
          asm_output
        | asm_output_list ',' asm_output
        ;

asm_inputs:
          ':' asm_input_list
        ;

asm_input:
          string '(' comma_expression ')'
        ;

asm_input_list:
          asm_input
        | asm_input_list ',' asm_input

asm_clobbered_registers:
          ':' asm_clobbered_registers_list
        ;

asm_clobbered_registers_list:
          string
        | asm_clobbered_registers_list ',' string
        ;

/*** External Definitions ***********************************************/


/* note: the following has been changed from the ANSI-C grammar:	*/
/*	- translation unit is allowed to be empty!			*/

translation_unit:
	/* nothing */
	| external_definition_list
	;

external_definition_list:
	external_definition
	| external_definition_list external_definition
	;

external_definition:
	function_definition
	| declaration
	| ';' // empty declaration
	;

function_definition:
	function_head KnR_parameter_header_opt compound_statement
	{ 
          stack($1).add("value").swap(stack($3));
          PARSER.pop_scope();
          PARSER.move_declaration(stack($1));
          PARSER.function="";
	}
	/* This is a GCC extension */
	| function_head KnR_parameter_header_opt gcc_asm_statement
	{ 
          // we ignore the value for now
          //stack($1).add("value").swap(stack($3));
          PARSER.pop_scope();
          PARSER.move_declaration(stack($1));
          PARSER.function="";
	}
	;

KnR_parameter_header_opt:
          /* empty */
	| KnR_parameter_header
	;

KnR_parameter_header:
	  KnR_parameter_declaration
	| KnR_parameter_header KnR_parameter_declaration
	;

KnR_parameter_declaration: declaring_list ';'
	;

function_head:
	identifier_declarator /* void */
	{
	  init($$);
	  irept type("int");
	  PARSER.new_declaration(type, stack($1), stack($$));
	  create_function_scope(stack($$));
	}
	| declaration_specifier declarator
	{
	  init($$);
	  PARSER.new_declaration(stack($1), stack($2), stack($$));
	  create_function_scope(stack($$));
	}
	| type_specifier declarator
	{
	  init($$);
	  PARSER.new_declaration(stack($1), stack($2), stack($$));
	  create_function_scope(stack($$));
	}
	| declaration_qualifier_list identifier_declarator
	{
	  init($$);
	  PARSER.new_declaration(stack($1), stack($2), stack($$));
	  create_function_scope(stack($$));
	}
	| type_qualifier_list identifier_declarator
	{
	  init($$);
	  PARSER.new_declaration(stack($1), stack($2), stack($$));
	  create_function_scope(stack($$));
	}
	;

declarator:
	identifier_declarator
	| typedef_declarator
	;

typedef_declarator:
	paren_typedef_declarator
	| parameter_typedef_declarator
	;

parameter_typedef_declarator:
	typedef_name
	| typedef_name postfixing_abstract_declarator
	{
	  $$=$1;
	  make_subtype($$, $2);
	}
	| clean_typedef_declarator
	;

clean_typedef_declarator:	/* Declarator */
	clean_postfix_typedef_declarator
	| '*' parameter_typedef_declarator
	{
	  $$=$2;
	  do_pointer($1, $2);
	}
	| '*' type_qualifier_list parameter_typedef_declarator
	{
	  merge_types($2, $3);
	  $$=$2;
	  do_pointer($1, $2);
	}
	;

clean_postfix_typedef_declarator:	/* Declarator */
	'(' clean_typedef_declarator ')'
	{ $$ = $2; }
	| '(' clean_typedef_declarator ')' postfixing_abstract_declarator
	{
	  /* note: this is a pointer ($2) to a function ($4) */
	  /* or an array ($4)! */
	  $$=$2;
	  make_subtype($$, $4);
	}
	;

paren_typedef_declarator:	/* Declarator */
	paren_postfix_typedef_declarator
	| '*' '(' simple_paren_typedef_declarator ')'
	{
	  $$=$3;
	  do_pointer($1, $3);
	}
	| '*' type_qualifier_list '(' simple_paren_typedef_declarator ')'
	{
	  // not sure where the type qualifiers belong
	  merge_types($2, $4);
	  $$=$2;
	  do_pointer($1, $2);
	}
	| '*' paren_typedef_declarator
	{
	  $$=$2;
	  do_pointer($1, $2);
	}
	| '*' type_qualifier_list paren_typedef_declarator
	{
	  merge_types($2, $3);
	  $$=$2;
	  do_pointer($1, $2);
	}
	;

paren_postfix_typedef_declarator:	/* Declarator */
	'(' paren_typedef_declarator ')'
	{ $$ = $2; }
	| '(' simple_paren_typedef_declarator postfixing_abstract_declarator ')'
	{	/* note: this is a function ($3) with a typedef name ($2) */
	  $$=$2;
	  make_subtype($$, $3);
	}
	| '(' paren_typedef_declarator ')' postfixing_abstract_declarator
	{
	  /* note: this is a pointer ($2) to a function ($4) */
	  /* or an array ($4)! */
	  $$=$2;
	  make_subtype($$, $4);
	}
	;

simple_paren_typedef_declarator:
	typedef_name
	{
	  assert(0);
	}
	| '(' simple_paren_typedef_declarator ')'
	{ $$ = $2; }
	;

identifier_declarator:
	pointer_identifier_declarator direct_identifier_declarator
	{
	  // Merge some types picked up in pointer_... with the direct_ident...
	  // generated declarator. It may already have a type.
	  make_subtype($2, $1);
	  $$ = $2;
	}
	| direct_identifier_declarator
	;

pointer_identifier_declarator:
	'*'
	{
	  newstack($$);
	  do_pointer($1, $$);
	  $$ = $1;
	}
	| '*' type_qualifier_list
	{
	  do_pointer($1, $2);
	  $$ = $1;
	}
	| '*' type_qualifier_list pointer_identifier_declarator
	{
	  do_pointer($1, $3);
	  merge_types($1, $2);
	  $$ = $1;
	}
	;

direct_identifier_declarator:
	identifier
	{
	  // All identifier_declarators are based from this.
	  newstack($$);
	  stack($$).id("declarator");
	  stack($$).add("identifier") = stack($1);
	}
	| '(' identifier_declarator ')'
	{
		$$ = $2;
	}
	| direct_identifier_declarator postfixing_abstract_declarator
	{
		$$ = $1;
		make_subtype($$, $2);
	}

abstract_declarator:
	unary_abstract_declarator
	| postfix_abstract_declarator
	| postfixing_abstract_declarator
	;

parameter_abstract_declarator:
	parameter_unary_abstract_declarator
	| parameter_postfix_abstract_declarator
	;

postfixing_abstract_declarator:	/* AbstrDeclarator */
	array_abstract_declarator
	| '(' ')'
	{
	  $$=$1;
	  set($$, "code");
	  stack($$).add("arguments");
	  stack($$).add("subtype").make_nil();
	}
	| '('
	  {
		unsigned prefix=++PARSER.current_scope().compound_counter;
		PARSER.new_scope(i2string(prefix)+"::");
	  }
	  parameter_type_list ')'
	{
	  $$=$1;
	  set($$, "code");
	  stack($$).add("subtype").make_nil();
	  stack($$).add("arguments").get_sub().
	    swap(stack($3).add("subtypes").get_sub());
	  PARSER.pop_scope();
	}
	;

parameter_postfixing_abstract_declarator:
	array_abstract_declarator
	| '(' ')'
	{
	  $$=$1;
	  set($$, "code");
	  stack($$).add("arguments");
	  stack($$).add("subtype").make_nil();
	}
	| '('
	  {
		unsigned prefix=++PARSER.current_scope().compound_counter;
		PARSER.new_scope(i2string(prefix)+"::");
	  }
	  parameter_type_list ')'
	{
	  $$=$1;
	  set($$, "code");
	  stack($$).add("subtype").make_nil();
	  stack($$).add("arguments").get_sub().
	    swap(stack($3).add("subtypes").get_sub());
	  PARSER.pop_scope();
	}
	;

array_abstract_declarator:
	'[' ']'
	{
	  $$=$1;
	  set($$, "incomplete_array");
	  stack($$).add("subtype").make_nil();
	}
	| '[' constant_expression ']'
	{
	  $$=$1;
	  set($$, "array");
	  stack($$).add("size").swap(stack($2));
	  stack($$).add("subtype").make_nil();
	}
	| array_abstract_declarator '[' constant_expression ']'
	{
	  // we need to push this down
	  $$=$1;
	  set($2, "array");
	  stack($2).add("size").swap(stack($3));
	  stack($2).add("subtype").make_nil();
	  make_subtype($1, $2);
	}
	;

unary_abstract_declarator:
	'*'
	{
	  $$=$1;
	  set($$, "pointer");
	  stack($$).add("subtype").make_nil();
	}
	| '*' type_qualifier_list
	{
	  $$=$2;
	  exprt nil_declarator(static_cast<const exprt &>(get_nil_irep()));
	  merge_types(stack($2), nil_declarator);
	  do_pointer($1, $2);
	}
	| '*' abstract_declarator
	{
	  $$=$2;
	  do_pointer($1, $2);
	}
	| '*' type_qualifier_list abstract_declarator
	{
	  $$=$2;
	  merge_types($2, $3);
	  do_pointer($1, $2);
	}
	;

parameter_unary_abstract_declarator:
	'*'
	{
          $$=$1;
          set($$, "pointer");
          stack($$).add("subtype").make_nil();
	}
	| '*' type_qualifier_list
	{
          $$=$2;
          exprt nil_declarator(static_cast<const exprt &>(get_nil_irep()));
          merge_types(stack($2), nil_declarator);
          do_pointer($1, $2);
	}
	| '*' parameter_abstract_declarator
	{
          $$=$2;
          do_pointer($1, $2);
	}
	| '*' type_qualifier_list parameter_abstract_declarator
	{
          $$=$2;
          merge_types($2, $3);
          do_pointer($1, $2);
	}
	;

postfix_abstract_declarator:
	'(' unary_abstract_declarator ')'
	{ $$ = $2; }
	| '(' postfix_abstract_declarator ')'
	{ $$ = $2; }
	| '(' postfixing_abstract_declarator ')'
	{ $$ = $2; }
	| '(' unary_abstract_declarator ')' postfixing_abstract_declarator
	{
	  /* note: this is a pointer ($2) to a function ($4) */
	  /* or an array ($4) of pointers with name ($2)! */
	  $$=$2;
	  make_subtype($$, $4);
	}
	;

parameter_postfix_abstract_declarator:
	'(' parameter_unary_abstract_declarator ')'
	{ $$ = $2; }
	| '(' parameter_postfix_abstract_declarator ')'
	{ $$ = $2; }
	| parameter_postfixing_abstract_declarator
	| '(' parameter_unary_abstract_declarator ')' parameter_postfixing_abstract_declarator
	{
	  /* note: this is a pointer ($2) to a function ($4) */
	  /* or an array ($4) of pointers with name ($2)! */
	  $$=$2;
	  make_subtype($$, $4);
	}
	;

%%
