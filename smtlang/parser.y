%{
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include <expr.h>

#include "smt_parser.h"
#include "expr2smt.h"

#define YYDEBUG 1
#define YYINITDEPTH 10000
#define YYSTYPE unsigned

#define mto(x, y) stack(x).move_to_operands(stack(y))

extern char *yysmttext;

static short *malloced_yyss = NULL;
static void *malloced_yyvs = NULL;

#define yyoverflow( MSG, SS, SSSIZE, VS, VSSIZE, YYSSZ )   \
  do {                                                     \
   size_t newsize;                                         \
   short *newss;                                           \
   YYSTYPE *newvs;                                         \
   newsize = *(YYSSZ) *= 2;                                \
   if (malloced_yyss)                                      \
   {                                                       \
      newss = (short *) realloc (*(SS), newsize * sizeof (short));  \
      newvs = (YYSTYPE *) realloc (*(VS), newsize * sizeof (YYSTYPE));  \
   }                                                       \
   else                                                    \
   {                                                       \
      newss = (short *) malloc (newsize * sizeof (short));  \
      if (newss)                                           \
        memcpy (newss, *(SS), (SSSIZE));                   \
      newvs = (YYSTYPE *) malloc (newsize * sizeof (YYSTYPE)); \
      if (newvs)                                           \
        memcpy (newvs, *(VS), (VSSIZE));                   \
   }                                                       \
   if (!newss || !newvs)                                   \
   {                                                       \
     yyerror (MSG);                                        \
     return 2;                                             \
   }                                                       \
   *(SS) = malloced_yyss = newss;                          \
   malloced_yyvs = *(VS) = newvs;                          \
 } while (0);


void free_parser_stacks ( void )
{
  // std::cout << "Freeing parser stack." << std::endl;
  if (malloced_yyss)
  {
    free (malloced_yyss);
    free (malloced_yyvs);
    malloced_yyvs = malloced_yyss = NULL;
  }
}

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

static void init(YYSTYPE &expr, const irep_idt &id)
{
  init(expr);
  stack(expr).id(id);
}

/*******************************************************************\

Function: yysmterror

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

int yysmterror(const char *error)
{
  PARSER.parse_error(error, yysmttext);
  return strlen(error)+1;
}

int yysmtlex();

%}


%token NOT AND OR IMPLIES XOR IFF IF_THEN_ELSE
%token IDENT NUMERAL STATUS_ID RATIONAL INDEX
%token LPAR RPAR LBR RBR LCB RCB 
%token STATUS BENCHMARK LOGICA EXTRAFUNS EXTRAPREDS EXTRASORTS
%token ASSUMPTION FORMULA TRUE FALSE DISTINCT LET FLET ITE VAR FVAR
%token EQ LT GT PLUS MINUS MULT DIV MOD ANOT LEQ GEQ
%token FORALL EXISTS ARITH_SYMB
%token NOTES QUOTE STRING_CONTENT ATTRIBUTE USER_VALUE_CONTENT
%token THEORY THEORYA SORTS FUNS PREDS AXIOMS DEFINITION
%token LANGUAGE EXTENSIONS LOGIC

%start smt

%%

smt: benchmark | theory | logic ;      

benchmark: LPAR BENCHMARK
             { smt_parse_treet::benchmarkt newbench;
               PARSER.parse_tree.benchmarks.push_back(newbench); }
           bench_name 
             { PARSER.parse_tree.benchmarks.back().name = 
                stack($3);
               PARSER.parse_tree.benchmarks.back().location = 
                stack($3).find_location(); }
           bench_attributes RPAR;                            

bench_name: ident;

bench_attributes: 
      bench_attributes bench_attribute
    | bench_attribute
    ;

bench_attribute: 
      LOGICA logic_name 
      { PARSER.parse_tree.benchmarks.back().logics.push_back(stack($2)); }
    | STATUS status
      { smt_parse_treet::benchmarkt &b = PARSER.parse_tree.benchmarks.back(); 
        if(stack($2).id()=="sat") 
          b.status.push_back(smt_parse_treet::benchmarkt::SAT); 
        else if(stack($2).id()=="unsat") 
          b.status.push_back(smt_parse_treet::benchmarkt::UNSAT);
        else
          b.status.push_back(smt_parse_treet::benchmarkt::UNKNOWN);
      }
    | EXTRAFUNS LPAR fun_symb_decls RPAR 
      { 
        forall_operands(it, stack($3)) {
          std::list<exprt> &fl = 
            PARSER.parse_tree.benchmarks.back().function_symbols[it->id_string()];
          fl.push_back(*it);
        }
        stack($3).clear();
      }
    | EXTRAPREDS LPAR pred_symb_decls RPAR 
      { 
        forall_operands(it, stack($3)) {
          std::list<exprt> &fl = 
            PARSER.parse_tree.benchmarks.back().predicate_symbols[it->id_string()];
          fl.push_back(*it);
        }
        stack($3).clear(); 
      }
    | ASSUMPTION an_formula
      { PARSER.parse_tree.benchmarks.back().assumptions.push_back(stack($2)); }
    | FORMULA an_formula
      { PARSER.parse_tree.benchmarks.back().formulas.push_back(stack($2)); }
    | NOTES string
      { PARSER.parse_tree.benchmarks.back().notes.push_back(stack($2).id_string()); }
    | EXTRASORTS LPAR sort_symbs RPAR
      { 
        forall_operands(it, stack($3)) {
          PARSER.parse_tree.benchmarks.back().sort_symbols.push_back(*it);
        }
        stack($3).clear();
      }
    | annotation 
      { PARSER.parse_tree.benchmarks.back().annotations.insert(
          std::pair<irep_idt, irep_idt>(
            stack($1).get("name"),
            stack($1).get("value") ));
      }
    ;

logic_name: ident;

status: STATUS_ID;

an_formulas:
      an_formulas an_formula    { mto($$, $2); }
    | an_formula                { init($$, "formulas"); mto($$, $1); }
    ;

an_formula: 
      an_atom 
    | LPAR connective an_formulas RPAR
      { $$=$2; 
        Forall_operands(oit, stack($3)) 
        {
          if( ( oit->id()=="and" ||
                oit->id()=="or" ||
                oit->id()=="xor" ||
                oit->id()=="iff"
                ) && 
                stack($$).id()==oit->id()) 
          {
            Forall_operands(ooit, *oit) 
            {
              stack($$).move_to_operands(*ooit);
            }
            oit->clear();
          } else {
            stack($$).move_to_operands(*oit);
          }
        } 
        stack($3).clear();
      }
    | LPAR connective an_formulas annotations RPAR 
      { $$=$2;
        Forall_operands(oit, stack($3)) 
        {
          if( ( oit->id()=="and" ||
                oit->id()=="or" ||
                oit->id()=="xor" ||
                oit->id()=="iff"
                ) && 
                stack($$).id()==oit->id()) 
          {
            Forall_operands(ooit, *oit) 
            {
              stack($$).move_to_operands(*ooit);
            }
            oit->clear();
          } else {
            stack($$).move_to_operands(*oit);
          }
          stack($3).clear();
        }
        stack($$).set("annotations", stack($4)); }
    | LPAR quant_symb quant_vars an_formula RPAR
      { $$=$2; 
        stack($$).set("qvars", stack($3)); 
        stack($$).move_to_operands(stack($4));
      }
    | LPAR quant_symb quant_vars an_formula annotations RPAR
      { $$=$2; 
        stack($$).set("qvars", stack($3)); 
        stack($$).move_to_operands(stack($4));
        stack($$).set("annotations", stack($5)); }
    | LPAR LET LPAR var an_term RPAR an_formula RPAR 
      { init($$, "let");
        exprt &vars = static_cast<exprt&>(stack($$).add("variables"));        
        exprt &terms = static_cast<exprt&>(stack($$).add("vterms"));        
        if (stack($7).id()=="let") {
          exprt &v = static_cast<exprt&>(stack($7).add("variables"));
          exprt &t = static_cast<exprt&>(stack($7).add("vterms"));
          
          assert(v.operands().size()>0);          
          vars.swap(v); // move the whole thing
          terms.swap(t);
          
          assert(stack($7).operands().size()==1);
          stack($$).move_to_operands(stack($7).op0());
        } else
          stack($$).move_to_operands(stack($7));
        
        // the order of these declarations will be reversed!
        vars.move_to_operands(stack($4));
        terms.move_to_operands(stack($5));
      }
    | LPAR LET LPAR var an_term RPAR an_formula annotations RPAR 
      { init($$, "let"); 
        exprt &vars = static_cast<exprt&>(stack($$).add("variables"));
        exprt &terms = static_cast<exprt&>(stack($$).add("vterms"));        
        if (stack($7).id()=="let") {
          exprt &v = static_cast<exprt&>(stack($7).add("variables"));
          exprt &t = static_cast<exprt&>(stack($7).add("vterms"));
          
          assert(v.operands().size()>0);          
          vars.swap(v); // move the whole thing
          terms.swap(t);
          
          assert(stack($7).operands().size()==1);
          stack($$).move_to_operands(stack($7).op0());
        } else
          stack($$).move_to_operands(stack($7));
        
        // the order of these declarations will be reversed!
        vars.move_to_operands(stack($4));
        terms.move_to_operands(stack($5));
        stack($$).set("annotations", stack($8)); 
      } 
    | LPAR FLET LPAR fvar an_formula RPAR an_formula RPAR
      { init($$, "flet"); 
        exprt &vars = static_cast<exprt&>(stack($$).add("variables"));
        exprt &formulas = static_cast<exprt&>(stack($$).add("vformulas"));        
        if (stack($7).id()=="flet") {
          exprt &v = static_cast<exprt&>(stack($7).add("variables"));
          exprt &f = static_cast<exprt&>(stack($7).add("vformulas"));
          
          assert(v.operands().size()>0);          
          vars.swap(v); // move the whole thing
          formulas.swap(f);
          v.clear();
          f.clear();
                    
          assert(stack($7).operands().size()==1);
          stack($$).move_to_operands(stack($7).op0());
        } else
          stack($$).move_to_operands(stack($7));
        
        // the order of these declarations will be reversed!
        vars.move_to_operands(stack($4));
        formulas.move_to_operands(stack($5));
      } 
    | LPAR FLET LPAR fvar an_formula RPAR an_formula annotations RPAR
      { init($$, "flet"); 
        exprt &vars = static_cast<exprt&>(stack($$).add("variables"));
        exprt &formulas = static_cast<exprt&>(stack($$).add("vformulas"));        
        if (stack($7).id()=="flet") {
          exprt &v = static_cast<exprt&>(stack($7).add("variables"));
          exprt &f = static_cast<exprt&>(stack($7).add("vformulas"));
          
          assert(v.operands().size()>0);          
          vars.swap(v); // move the whole thing
          formulas.swap(f);
                    
          assert(stack($7).operands().size()==1);
          stack($$).move_to_operands(stack($7).op0());
        } else
          stack($$).move_to_operands(stack($7));
          
        // the order of these declarations will be reversed!
        vars.move_to_operands(stack($4));
        formulas.move_to_operands(stack($5));          
        stack($$).set("annotations", stack($8)); 
      } 
    ;

quant_symb:
      FORALL { init($$, "forall"); }
    | EXISTS { init($$, "exists"); }
    
quant_vars:
      quant_vars quant_var  { mto($$, $2); }
    | quant_var             { init($$, "qvars"); mto($$, $1); }
    ;
    
quant_var: LPAR var sort_symb RPAR { $$=$2;
                                     stack($$).add("type")=stack($3); }
    ;

connective: 
      NOT           { init($$, "not"); }
    | IF_THEN_ELSE  { init($$, "if_then_else"); }
    | IMPLIES       { init($$, "impl"); } 
    | AND           { init($$, "and"); }
    | OR            { init($$, "or"); }
    | XOR           { init($$, "xor"); }
    | IFF           { init($$, "iff"); }
    ;

fun_symb_decls: 
      fun_symb_decls fun_symb_decl { mto($$, $2); }
    | fun_symb_decl { init($$, "function_decls"); mto($$, $1); }
    ;

fun_symb_decl: 
      LPAR fun_symb sort_symbs RPAR { $$=$2; mto($$, $3); }
    | LPAR fun_symb sort_symbs annotations RPAR 
      { $$=$2; mto($$, $3); stack($$).set("annotations", stack($4)); }
    /* The following four rules are nonstandard, but suggested by C. Tinelli */
    | LPAR numeral sort_symbs RPAR { $$=$2; mto($$, $3); }
    | LPAR numeral sort_symbs annotations RPAR
      { $$=$2; mto($$, $3); stack($$).set("annotations", stack($4)); }
    | LPAR rational sort_symbs RPAR { $$=$2; mto($$, $3); }
    | LPAR rational sort_symbs annotations RPAR
      { $$=$2; mto($$, $3); stack($$).set("annotations", stack($4)); }
    ;

fun_symb: 
      ident { init($$, ii2string(stack($1))); }
    | ar_symb
    ;

ar_symb: ARITH_SYMB;

sort_symbs: 
      sort_symbs sort_symb  { mto($$, $2); }
    | sort_symb             { init($$, "sorts"); mto($$, $1); }
    ;

sort_symb: ident            { init($$, "sort"); 
                              stack($$).type()=typet(ii2string(stack($1))); }
    ;

pred_symb_decls: 
      pred_symb_decls pred_symb_decl { mto($$, $2); }
    | pred_symb_decl { init($$, "pred_decls"); mto($$, $1); }
    ;

pred_symb_decl: 
      LPAR pred_symb annotations RPAR 
      { $$=$2; stack($$).set("annotations", stack($3)); }
    | LPAR pred_symb sort_symbs annotations RPAR 
      { $$=$2; mto($$, $3); stack($$).set("annotations", stack($4)); }
    | LPAR pred_symb sort_symbs RPAR { $$=$2; mto($$, $3); }
    | LPAR pred_symb RPAR { $$=$2; }
    ;

pred_symb: 
      ident    { PARSER.set_location(stack($$)); 
                 init($$, ii2string(stack($1))); }
    | ar_symb  { PARSER.set_location(stack($$)); }
    | DISTINCT { init ($$, "distinct"); } 
    ;

an_atom: 
      prop_atom      
    | LPAR prop_atom RPAR { $$=$2; }
    | LPAR prop_atom annotations RPAR 
      { $$=$2; stack($$).set("annotations", stack($3)); }
    | LPAR pred_symb an_terms RPAR 
      { $$=$3;
        stack($$).id(stack($2).id());
      }
    | LPAR pred_symb an_terms annotations RPAR 
      { $$=$3; 
        stack($$).id(stack($2).id());
        stack($$).set("annotations", stack($4));; 
      }
    ;

an_terms: 
      an_terms an_term  { mto($$, $2); } 
    | an_term  { init($$, "terms"); mto($$, $1); }
    ;

an_term:
      base_term  
    | LPAR base_term annotations RPAR 
        { $$=$2; stack($$).set("annotations", stack($3)); }
    | LPAR fun_symb an_terms RPAR { $$=$3;
                                    stack($$).id(stack($2).id());
                                  }
    | LPAR fun_symb an_terms annotations RPAR 
                                  { $$=$3;
                                    stack($$).id(stack($2).id());
                                    stack($$).set("annotations", stack($4)); }
    | LPAR ITE an_formula an_term an_term RPAR 
                                  { init($$, "ite");
                                    mto($$, $3);
                                    mto($$, $4);
                                    mto($$, $5); }
    | LPAR ITE an_formula an_term an_term annotations RPAR 
                                  { init($$, "ite");
                                    mto($$, $3);
                                    mto($$, $4);
                                    mto($$, $5); 
                                    stack($$).set("annotations", stack($6)); }
    ;
    
annotations: 
      annotations annotation { mto($$,$2); }
    | annotation  { init($$, "annotations"); mto($$, $1); }
    ;
        
annotation: 
      attribute 
    | attribute LCB user_value RCB 
      { $$=$1; stack($$).set("value", stack($1).get("value")); }
    /* The following rule is nonstandard, but we need this for the
       theory files (e.g., BitVector32.smt) */
    | attribute string 
      { $$=$1; stack($$).set("value", stack($1).get("value")); }
    ;
    
attribute: ATTRIBUTE   { init($$, "attribute");
                         stack($$).set("name", stack($1).id_string()); }
    ;

user_value: USER_VALUE_CONTENT { init($$, "user_value"); 
                            stack($$).set("value", stack($1).id_string()); }
    ;

base_term: 
      var 
    | ident 
    | numeral
    | rational
    ;

var: VAR    { init($$, "var"); 
              stack($$).set("identifier", stack($1).id_string().substr(1)); }
    ;

fvar: FVAR  { init($$, "fvar"); 
              stack($$).set("identifier", stack($1).id_string().substr(1)); }
    ;

prop_atom: 
      TRUE  { init($$, "bool"); stack($$).set("value", "true"); }
    | FALSE { init($$, "bool"); stack($$).set("value", "false"); }
    | fvar
    | ident
    ;
    
simple_ident: IDENT;

index_ident: 
  ident index
  { $$=$1; mto($$, $2); };
    
ident: 
      simple_ident 
    | index_ident ;

numeral: NUMERAL;

rational: RATIONAL;

index: INDEX;

string: QUOTE STRING_CONTENT QUOTE;

theory: LPAR THEORY 
        { smt_parse_treet::theoryt newtheory;
          PARSER.parse_tree.theories.push_back(newtheory); }
        theory_name
        { PARSER.parse_tree.theories.back().name = 
            stack($3); 
          PARSER.parse_tree.theories.back().location = 
            stack($3).find_location(); }
        theory_attributes RPAR
    ;

theory_name: ident;

theory_attributes: 
      theory_attributes theory_attribute
    | theory_attribute
    ;

theory_attribute:
      SORTS LPAR sort_symbs RPAR 
      { PARSER.parse_tree.theories.back().sorts.push_back(stack($3)); }
    | FUNS LPAR fun_symb_decls RPAR
      { 
        std::list<exprt> &fl = 
          PARSER.parse_tree.theories.back().funs[stack($3).id_string()];
        fl.push_back(stack($3)); 
      }
    | PREDS LPAR pred_symb_decls RPAR
      { 
        std::list<exprt> &fl = 
          PARSER.parse_tree.theories.back().preds[stack($3).id_string()]; 
        fl.push_back(stack($3)); 
      }
    | DEFINITION string
      { 
        PARSER.parse_tree.theories.back().definitions.push_back(stack($2).id_string()); 
      }
    | AXIOMS LPAR an_formulas RPAR
      { PARSER.parse_tree.theories.back().axioms.push_back(stack($3)); }
    | NOTES string
      { PARSER.parse_tree.theories.back().notes.push_back(stack($2).id_string()); }
    | annotation
      { PARSER.parse_tree.theories.back().annotations.insert(
          std::pair<irep_idt, irep_idt>(
            stack($1).get("name"),
            stack($1).get("value") ));
      }
    ;
      
logic: LPAR LOGIC
       { smt_parse_treet::logict newlogic;          
         PARSER.parse_tree.logics.push_back(newlogic); }
       logic_name
       { PARSER.parse_tree.logics.back().name = 
          stack($3); 
         PARSER.parse_tree.logics.back().location = 
          stack($3).find_location(); }
       logic_attributes RPAR
    ;
       
logic_attributes:
      logic_attributes logic_attribute
    | logic_attribute
    ;
    
logic_attribute:  
      THEORYA theory_name
      { PARSER.parse_tree.logics.back().theories.push_back(stack($2)); }
    | LANGUAGE string
      { PARSER.parse_tree.logics.back().languages.push_back(stack($2).id_string()); }
    | EXTENSIONS string
      { PARSER.parse_tree.logics.back().extensions.push_back(stack($2).id_string()); }
    | NOTES string
      { PARSER.parse_tree.logics.back().notes.push_back(stack($2).id_string()); }
    | annotation
      { PARSER.parse_tree.logics.back().annotations.insert(
          std::pair<irep_idt, irep_idt>(
            stack($1).get("name"),
            stack($1).get("value") ));
      }
    ;

%%
