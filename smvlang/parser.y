%{
#include <i2string.h>

#include "smv_parser.h"
#include "smv_typecheck.h"

#define YYSTYPE unsigned
#define PARSER smv_parser

#include "y.tab.h"

#define YYMAXDEPTH 200000
#define YYSTYPE_IS_TRIVIAL 1

/*------------------------------------------------------------------------*/

#define yylineno yysmvlineno
#define yytext yysmvtext

#define yyerror yysmverror
int yysmverror(const std::string &error);
int yylex();
extern char *yytext;

/*------------------------------------------------------------------------*/

#define mto(x, y) stack(x).move_to_operands(stack(y))
#define binary(x, y, id, z) { init(x, id); \
  stack(x).move_to_operands(stack(y), stack(z)); }

/*******************************************************************\

Function: init

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void init(exprt &expr)
{
  expr.clear();
  PARSER.set_location(expr);
}

/*******************************************************************\

Function: init

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void init(YYSTYPE &expr)
{
  newstack(expr);
  init(stack(expr));
}

/*******************************************************************\

Function: init

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void init(YYSTYPE &expr, const std::string &id)
{
  init(expr);
  stack(expr).id(id);
}

/*******************************************************************\

Function: mk_index

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

#if 0
static void mk_index(YYSTYPE &dest, YYSTYPE &op, YYSTYPE &index)
{
  init(dest, "extractbit");
  stack(dest).set("index", stack(index));
  mto(dest, op);
}
#endif

/*******************************************************************\

Function: j_binary

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void j_binary(YYSTYPE &dest, YYSTYPE &op1,
                     const std::string &id, YYSTYPE &op2)
{
  if(stack(op1).id()==id)
  {
    dest=op1;
    mto(dest, op2);
  }
  else if(stack(op2).id()==id)
  {
    dest=op2;
    mto(dest, op1);
  }
  else
    binary(dest, op1, id, op2);
}

/*******************************************************************\

Function: new_module

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void new_module(YYSTYPE &module)
{
  const std::string name=smv_module_symbol(stack(module).id_string());
  PARSER.module=&PARSER.parse_tree.modules[name];
  PARSER.module->name=name;
  PARSER.module->base_name=stack(module).id_string();
  PARSER.module->used=true;
}

/*------------------------------------------------------------------------*/

%}

%token AG_Token AX_Token AF_Token

%token INIT_Token TRANS_Token SPEC_Token VAR_Token DEFINE_Token ASSIGN_Token
%token INVAR_Token FAIRNESS_Token MODULE_Token ARRAY_Token OF_Token
%token DOTDOT_Token BOOLEAN_Token EXTERN_Token

%token NEXT_Token INC_Token DEC_Token CASE_Token ESAC_Token BECOMES_Token
%token ADD_Token SUB_Token SWITCH_Token init_Token PLUS_Token

%token STRING_Token QSTRING_Token QUOTE_Token
%token NUMBER_Token

%right IMPLIES_Token
%left  UNION_Token
%left  EQUIV_Token
%left  XOR_Token
%left  OR_Token
%left  AND_Token
%left  NOT_Token
%left  EX_Token AX_Token EF_Token AF_Token EG_Token AG_Token E_Token A_Token UNTIL_Token
%left  EQUAL_Token NOTEQUAL_Token LT_Token GT_Token LE_Token GE_Token
%left  PLUS_Token MINUS_Token
%left  TIMES_Token DIVIDE_Token
%left  UMINUS           /* supplies precedence for unary minus */
%left  DOT_Token

%%

start      : modules
           | formula { PARSER.module->add_spec(stack($1));
                       PARSER.module->used=true; }
           ;

modules    : module
           | modules module
           ;

module     : module_head sections
           ;

module_name: STRING_Token
           | QUOTE_Token
           ;

module_head: MODULE_Token module_name { new_module($2); }
           | MODULE_Token module_name { new_module($2); } '(' module_argument_list_opt ')'
           ;

sections   : /* epsilon */
           | section sections
           ;

section    : VAR_Token vardecls
           | VAR_Token
           | INIT_Token formula         { PARSER.module->add_init(stack($2), stack($1).location()); }
           | INIT_Token formula ';'     { PARSER.module->add_init(stack($2), stack($1).location()); }
           | INIT_Token
           | TRANS_Token formula        { PARSER.module->add_trans(stack($2), stack($1).location()); }
           | TRANS_Token formula ';'    { PARSER.module->add_trans(stack($2), stack($1).location()); }
           | TRANS_Token
           | SPEC_Token formula         { PARSER.module->add_spec(stack($2), stack($1).location()); }
           | SPEC_Token formula ';'     { PARSER.module->add_spec(stack($2), stack($1).location()); }
           | SPEC_Token
           | ASSIGN_Token assignments
           | ASSIGN_Token
           | DEFINE_Token defines
           | DEFINE_Token
           | INVAR_Token formula        { PARSER.module->add_invar(stack($2), stack($1).location()); }
           | INVAR_Token formula ';'    { PARSER.module->add_invar(stack($2), stack($1).location()); }
           | INVAR_Token
           | FAIRNESS_Token formula     { PARSER.module->add_fairness(stack($2), stack($1).location()); }
           | FAIRNESS_Token formula ';' { PARSER.module->add_fairness(stack($2), stack($1).location()); }
           | FAIRNESS_Token
           | EXTERN_Token extern_var
           | EXTERN_Token extern_var ';'
           ;
 
extern_var : variable_name EQUAL_Token QUOTE_Token
{
  const irep_idt &identifier=stack($1).get("identifier");
  smv_parse_treet::mc_vart &var=PARSER.module->vars[identifier];

  if(var.identifier!="")
  {
    yyerror("variable `"+id2string(identifier)+"' already declared extern");
    YYERROR;
  }
  else
    var.identifier=stack($3).id_string();
}
           ;

vardecls   : vardecl
           | vardecl vardecls
           ;

module_argument: variable_name
{
  const irep_idt &identifier=stack($1).get("identifier");
  smv_parse_treet::mc_vart &var=PARSER.module->vars[identifier];
  var.var_class=smv_parse_treet::mc_vart::ARGUMENT;
  PARSER.module->ports.push_back(identifier);
}
           ;

module_argument_list: module_argument
           | module_argument_list ',' module_argument
           ;

module_argument_list_opt: /* empty */
           | module_argument_list
           ;

type       : ARRAY_Token NUMBER_Token DOTDOT_Token NUMBER_Token OF_Token type
{
  init($$, "array");
  int start=atoi(stack($2).id().c_str());
  int end=atoi(stack($4).id().c_str());

  if(end < start)
  {
    yyerror("array must end with number >= `"+i2string(start)+"'");
    YYERROR;
  }

  stack($$).set("size", end-start+1);
  stack($$).set("offset", start);
  stack($$).set("subtype", stack($6));
}
           | BOOLEAN_Token { init($$, "bool"); }
           | '{' enum_list '}' { $$=$2; }
           | NUMBER_Token DOTDOT_Token NUMBER_Token
            {
              init($$, "range");
              stack($$).set("from", stack($1));
              stack($$).set("to", stack($3));
            }
           | usertype
           ;

usertype   : module_name
            {
              init($$, "submodule");
              stack($$).set("identifier",
                            smv_module_symbol(stack($1).id_string()));
            }
           | module_name '(' formula_list ')'
            {
              init($$, "submodule");
              stack($$).set("identifier",
                            smv_module_symbol(stack($1).id_string()));
              stack($$).operands().swap(stack($3).operands());
            }
           ;

enum_list  : enum_element
              {
               init($$, "enum");
               stack($$).add("elements").get_sub().push_back(irept(stack($1).id()));
              }
           | enum_list ',' enum_element
              {
               $$=$1;
               stack($$).add("elements").get_sub().push_back(irept(stack($3).id())); 
              }
           ;

enum_element: STRING_Token
              {
                $$=$1;
                PARSER.module->enum_set.insert(stack($1).id_string());
              }
            ;

vardecl    : variable_name ':' type ';'
{
  const irep_idt &identifier=stack($1).get("identifier");
  smv_parse_treet::mc_vart &var=PARSER.module->vars[identifier];

  switch(var.var_class)
  {
  case smv_parse_treet::mc_vart::UNKNOWN:
    var.type=(typet &)stack($3);
    var.var_class=smv_parse_treet::mc_vart::DECLARED;
    break;

  case smv_parse_treet::mc_vart::DEFINED:
    yyerror("variable `"+id2string(identifier)+"' already defined");
    YYERROR;
    break;

  case smv_parse_treet::mc_vart::DECLARED:
    yyerror("variable `"+id2string(identifier)+"' already declared as variable");
    YYERROR;
    break;
  
  case smv_parse_treet::mc_vart::ARGUMENT:
    yyerror("variable `"+id2string(identifier)+"' already declared as argument");
    YYERROR;
    break;
  
  default:
    assert(false);
  }
}
           ;

assignments: assignment
           | assignment assignments
           | define
           | define assignments
           ;

assignment : assignment_head '(' assignment_var ')' BECOMES_Token formula ';'
{
  binary($$, $3, "=", $6);

  if(stack($1).id()=="next")
  {
    exprt &op=stack($$).op0();
    exprt tmp("smv_next");
    tmp.operands().resize(1);
    tmp.op0().swap(op);
    tmp.swap(op);
    PARSER.module->add_trans(stack($$));
  }
  else
    PARSER.module->add_init(stack($$));
}
;

assignment_var: variable_name
              ;

assignment_head: init_Token { init($$, "init"); }
               | NEXT_Token { init($$, "next"); }
               ;

defines: define
       | define defines
       ;

define : assignment_var BECOMES_Token formula ';'
{
  const irep_idt &identifier=stack($1).get("identifier");
  smv_parse_treet::mc_vart &var=PARSER.module->vars[identifier];

  switch(var.var_class)
  {
  case smv_parse_treet::mc_vart::UNKNOWN:
    var.type.make_nil();
    var.var_class=smv_parse_treet::mc_vart::DEFINED;
    break;

  case smv_parse_treet::mc_vart::DECLARED:
    var.var_class=smv_parse_treet::mc_vart::DEFINED;
    break;

  case smv_parse_treet::mc_vart::DEFINED:
    yyerror("variable `"+id2string(identifier)+"' already defined");
    YYERROR;
    break;
  
  case smv_parse_treet::mc_vart::ARGUMENT:
    yyerror("variable `"+id2string(identifier)+"' already declared as argument");
    YYERROR;
    break;
  
  default:
    assert(false);
  }

  binary($$, $1, "=", $3);
  PARSER.module->add_define(stack($$));
}
;

formula : term
        ;

term: variable_name
    | NEXT_Token '(' term ')' { init($$, "smv_next"); mto($$, $3); }
    | '(' formula ')' { $$=$2; }
    | '{' formula_list '}' { $$=$2; stack($$).id("smv_nondet_choice"); }
    | INC_Token '(' term ')' { init($$, "inc"); mto($$, $3); }
    | DEC_Token '(' term ')' { init($$, "dec"); mto($$, $3); }
    | ADD_Token '(' term ',' term ')' { j_binary($$, $3, "+", $5); }
    | SUB_Token '(' term ',' term ')' { init($$, "-"); mto($$, $3); mto($$, $5); }
    | NUMBER_Token { init($$, "number_constant"); stack($$).set("value", stack($1).id()); }
    | CASE_Token cases ESAC_Token { $$=$2; }
    | SWITCH_Token '(' variable_name ')' '{' switches '}' { init($$, "switch"); mto($$, $3); mto($$, $6); }
    | MINUS_Token term %prec UMINUS { init($$, "unary-"); mto($$, $2); }
    | term PLUS_Token term    { j_binary($$, $1, "+", $3); }
    | term MINUS_Token term   { j_binary($$, $1, "-", $3); }
    | term EQUIV_Token term   { binary($$, $1, "=", $3); }
    | term IMPLIES_Token term { binary($$, $1, "=>", $3); }
    | term XOR_Token term     { j_binary($$, $1, "xor", $3); }
    | term OR_Token term      { j_binary($$, $1, "or", $3); }
    | term AND_Token term     { j_binary($$, $1, "and", $3); }
    | NOT_Token term { init($$, "not"); mto($$, $2); }
    | AX_Token  term { init($$, "AX");  mto($$, $2); }
    | AF_Token  term { init($$, "AF");  mto($$, $2); }
    | AG_Token  term { init($$, "AG");  mto($$, $2); }
    | term EQUAL_Token    term { binary($$, $1, "=",  $3); }
    | term NOTEQUAL_Token term { binary($$, $1, "notequal", $3); }
    | term LT_Token       term { binary($$, $1, "<",  $3); }
    | term LE_Token       term { binary($$, $1, "<=", $3); }
    | term GT_Token       term { binary($$, $1, ">",  $3); }
    | term GE_Token       term { binary($$, $1, ">=", $3); }
    | term UNION_Token    term { binary($$, $1, "smv_union", $3); }
    ;

formula_list: formula { init($$); mto($$, $1); }
            | formula_list ',' formula { $$=$1; mto($$, $3); }
            ;

variable_name: qstring_list
		{
                 const std::string &id=stack($1).id_string();

                 bool is_enum=(PARSER.module->enum_set.find(id)!=
                               PARSER.module->enum_set.end());
                 bool is_var=(PARSER.module->vars.find(id)!=
                              PARSER.module->vars.end());

                 if(is_var && is_enum)
                  {
                   yyerror("identifier `"+id+"' is ambiguous");
                   YYERROR;
                  }
                 else if(is_enum)
                  {
                   init($$, "enum_constant");
                   stack($$).type()=typet("enum");
                   stack($$).set("value", stack($1).id());
                  }
                 else // not an enum, probably a variable
                  {
                   init($$, "symbol");
                   stack($$).set("identifier", stack($1).id());
                   //PARSER.module->vars[stack($1).id()];
                  }
                }
             ;

qstring_list: QSTRING_Token
              {
                init($$, std::string(stack($1).id_string(), 1)); // remove backslash
              }
             | STRING_Token
             | qstring_list DOT_Token QSTRING_Token
              {
                std::string id(stack($1).id_string());
                id+=".";
                id+=std::string(stack($3).id_string(), 1); // remove backslash
                init($$, id);
              }
             | qstring_list DOT_Token STRING_Token
              {
                std::string id(stack($1).id_string());
                id+=".";
                id+=stack($3).id_string();
                init($$, id);
              }
             | qstring_list '[' NUMBER_Token ']'
              {
                std::string id(stack($1).id_string());
                id+="[";
                id+=stack($3).id_string();
                id+="]";
                init($$, id);
              }
             | qstring_list '(' NUMBER_Token ')'
              {
                std::string id(stack($1).id_string());
                id+="(";
                id+=stack($3).id_string();
                id+=")";
                init($$, id);
              }
             ;

cases   : { init($$, "smv_cases"); }
        | cases case { $$=$1; mto($$, $2); }
        ;

case    : formula ':' formula ';' { binary($$, $1, "case", $3); }
        ;

switches: { init($$, "switches"); }
        | switches switch { $$=$1; mto($$, $2); }
        ;

switch  : NUMBER_Token ':' term ';' { init($$, "switch"); mto($$, $1); mto($$, $3); }
        ;

%%

int yysmverror(const std::string &error)
{
  PARSER.parse_error(error, yytext);
  return 0;
}

#undef yyerror

