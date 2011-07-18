#include <std_code.h>
#include <std_types.h>
#include <std_expr.h>
#include <expr_util.h>
#include <i2string.h>

#include <ansi-c/c_types.h>

#define YYSTYPE unsigned
#define YYSTYPE_IS_TRIVIAL 1

#define mto(x, y) stack(x).move_to_operands(stack(y))
#define binary(x, y, l, id, z) { init(x, id); \
  stack(x).location()=stack(l).location(); \
  stack(x).reserve_operands(2); mto(x, y); mto(x, z); }

static void mts(typet &x, typet &t)
{

  if (t.find("declarator").id() != "nil" && !t.find("declarator").is_nil()) {
    x.add("declarator") = t.find("declarator");
    t.remove("declarator");
  }

  x.move_to_subtypes(t);
  return;
}

static void mts(YYSTYPE x, YYSTYPE t)
{

  mts((typet&)stack(x),(typet&)stack(t));
  return;
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
}

/*******************************************************************\

Function: init

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void init(YYSTYPE &expr, const std::string &id)
{
  newstack(expr);
  stack(expr).id(id);
}

/*******************************************************************\

Function: set

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void set(YYSTYPE expr, const std::string &id)
{
  stack(expr).id(id);
}

/*******************************************************************\

Function: statement

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void statement(YYSTYPE &expr, const std::string &id)
{
  set(expr, "code");
  stack(expr).set("statement", id);
}

/*******************************************************************\

Function: decl_statement

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void decl_statement(
  YYSTYPE &dest,
  YYSTYPE &decl,
  YYSTYPE &initializer)
{
  codet decl_statement("decl");
  
  symbol_exprt symbol;
  symbol.set_identifier(stack(decl).get("name"));
  
  decl_statement.move_to_operands(symbol);
  
  if(stack(initializer).is_not_nil())
  {
    stack(decl).add("value")=stack(initializer);
    decl_statement.move_to_operands(stack(initializer));
  }
  
  PARSER.move_declaration(stack(decl));
  
  stack(dest).move_to_operands(decl_statement);
}

/*******************************************************************\

Function: merge_types

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void merge_types(exprt &dest, exprt &src)
{
  #if 0
  std::cout << "D: " << dest.pretty() << std::endl;
  std::cout << "S: " << src.pretty() << std::endl;
  #endif

  if(dest.id()!="merged_type")
  {
    locationt location=dest.location();
    typet new_type("merged_type");
    mts(new_type, (typet&)dest);
    dest.swap(new_type);
    dest.location()=location;
  }

  mts((typet&)dest, (typet&)src);
}

/*******************************************************************\

Function: merge_types

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void merge_types(const YYSTYPE dest, const YYSTYPE src)
{
  merge_types(stack(dest), stack(src));
}

/*******************************************************************\

Function: make_subtype

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void make_subtype(typet &dest, typet &src)
{  
  // inserts "src" into "dest"
  // e.g., src is a pointer or array,
  // dest is a symbol or type
  
  // find spot where to insert this
  
  #if 0
  std::cout << "D: " << dest.pretty() << std::endl;
  std::cout << "S: " << src.pretty() << std::endl;
  std::cout << "src.id(): " << src.id() << std::endl;
  #endif
  
  assert(src.id()=="array" ||
         src.id()=="incomplete_array" ||
         src.id()=="pointer" ||
         src.id()=="code" ||
         src.id()=="merged_type");

  typet *p=&dest;

  while(true)
  {
    // see if we need to walk down
    typet *sub=p;
    
    if(p->id()=="merged_type")
    {
      // do last one
      assert(!p->subtypes().empty());
      sub=&(p->subtypes().back());
    }

    if(sub->id()=="pointer" ||
       sub->id()=="array" ||
       sub->id()=="incomplete_array" ||
       sub->id()=="code")
    {
      // walk down
      p=&sub->subtype();    
    }
    else
    {
      if(p->is_nil())
      {
        p->swap(src);
        break;
      }
      else
      {
        // *p is now type or symbol
    
        // save symbol
        typet symbol=*p;
        p->swap(src);
      
        // find spot where to put symbol
        while(true)
        {
          if(p->is_nil())
          {
            p->swap(symbol);
            break;
          }
          else if(p->id()=="merged_type")
          {
            assert(!p->subtypes().empty());
            p=&(p->subtypes().back());
          }
          else if(p->id()=="")
            assert(false);
          else
            p=&p->subtype();
        }
        break;
      }
    }
  }
}

/*******************************************************************\

Function: make_subtype

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void make_subtype(YYSTYPE dest, YYSTYPE src)
{
  make_subtype((typet &)stack(dest), (typet &)stack(src));
}
    
/*******************************************************************\

Function: do_pointer

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static typet & find_sub_merged_type(typet &t)
{

  assert(t.id() == "merged_type");
  assert(t.subtypes().size() != 0);
  typet &end = t.subtypes().back();
  if (end.id() == "merged_type")
    return find_sub_merged_type(end);

  return end;
}

static void do_pointer(const YYSTYPE ptr, const YYSTYPE dest)
{
  typet *target;

  set(ptr, "pointer");

  if (stack(dest).id() == "merged_type")
    target = &find_sub_merged_type((typet&)stack(dest));
  else
    target = (typet*) &stack(dest);

  stack(ptr).add("subtype") = target->find("subtype");
  target->add("subtype") = stack(ptr);

}

static void insert_subtype(irept &target, const typet &type)
{

  if (target.id() == "merged_type") {
    insert_subtype(((typet&)target).subtypes().back(), type);
    return;
  }

  const irept &atype = target.find("subtype");
  if (atype.id() == "nil" || atype.is_nil()) {
    target.add("subtype") = type;
  } else {
    typet *wheretoadd = &(typet&)target.add("subtype");
    while (wheretoadd->find("subtype").id() != "nil" && !wheretoadd->find("subtype").is_nil()) {
      wheretoadd = (typet *)&wheretoadd->find("subtype");
    }
    wheretoadd->add("subtype") = type;
  }

  return;
}


static void insert_declarator(exprt& dest, exprt& src)
{

  irept *t = &src.add("subtype");
  if (t->is_nil() || t->id() == "") {
    t = &dest;
    t->add("declarator") = src;
  } else {
    insert_subtype(*t, (typet&)dest);
    t->add("declarator") = src;
    t->add("declarator").remove("subtype");
    dest = *(exprt*&)t;
  }

  return;
}

static void insert_declarator(YYSTYPE dest, YYSTYPE src)
{

  insert_declarator(stack(dest), stack(src));
  return;
}

/*******************************************************************\

Function: do_enum_members

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void do_enum_members(
  const typet &enum_type,
  exprt &members)
{  
  exprt value;

  // start with 0  
  value=gen_zero(int_type());

  Forall_operands(it, members)
  {
    it->type()=enum_type;
    
    exprt &v=(exprt &)it->add("value");
    if(v.is_nil())
      v=value;

    exprt symbol_expr("symbol");
    symbol_expr.set("identifier", it->get("name"));
    
    value=exprt("+");
    value.copy_to_operands(symbol_expr, gen_one(int_type()));    
    
    PARSER.copy_declaration(*it);
  }
}

/*******************************************************************\

Function: do_tag

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void do_tag(YYSTYPE &key, YYSTYPE &tag)
{
  irep_idt id_class_str=stack(tag).get("#id_class");

  ansi_c_id_classt id_class=
    (ansi_c_id_classt)atoi(id_class_str.c_str());

  if(id_class==ANSI_C_TAG) // we have it already
    return;

  // it's new
  ansi_c_declarationt declaration;
  PARSER.new_declaration(stack(key), stack(tag), declaration, true);

  // grab symbol
  stack(tag).id("symbol");
  stack(tag).set("identifier", declaration.get_name());
  stack(tag).location()=declaration.location();

  declaration.type().id("incomplete_"+declaration.type().id_string());
  PARSER.move_declaration(declaration);                    
}

/*******************************************************************\

Function: create_function_scope

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void create_function_scope(exprt &expr)
{
  ansi_c_declarationt &declaration=to_ansi_c_declaration(expr);

  PARSER.function=declaration.get_base_name();

  std::string prefix=PARSER.current_scope().prefix+
                     id2string(declaration.get_base_name())+"::";
  PARSER.new_scope(prefix);
  
  if(declaration.type().id()=="code")
  {
    code_typet &code_type=to_code_type(declaration.type());
    
    code_typet::argumentst &arguments=code_type.arguments();
    
    unsigned anon_count=0;
    
    // do the parameter declarations
    for(code_typet::argumentst::iterator
        it=arguments.begin();
        it!=arguments.end();
        it++)
    {
      if(it->id()=="declaration")
      {
        ansi_c_declarationt &arg_decl=to_ansi_c_declaration(*it);
        
        if(arg_decl.type().id()=="incomplete_array")
          arg_decl.type().id("pointer");

        if(arg_decl.get_base_name()=="")
          arg_decl.set_base_name("#anon_arg"+i2string(anon_count++));

        // make sure we know it's an argument
        arg_decl.set_is_argument(true);
          
        // fix name
        arg_decl.set_name(
          PARSER.current_scope().prefix+id2string(arg_decl.get_base_name()));
      
        // copy declaration
        PARSER.copy_declaration(arg_decl);
      
        // add to scope
        PARSER.current_scope().name_map
          [arg_decl.get_base_name()].id_class=ANSI_C_SYMBOL;
      }
    }
  }
}
