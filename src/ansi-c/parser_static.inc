#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/std_code.h>
#include <util/std_types.h>
#include <util/std_expr.h>

#define mto(x, y) (x)->move_to_operands(*(y))
#define mts(x, y) (x)->move_to_subtypes(*(y))
#define binary(x, y, l, id, z) { init(&(x), (id)); \
  (x)->location()=(l)->location(); \
  (x)->reserve_operands(2); mto((x), (y)); mto((x), (z)); }

static void init(exprt **e)
{
  *e = new exprt();
}

static void init(typet **e)
{
  *e = new typet();
}

static void init(ansi_c_declarationt **d)
{

  *d = new ansi_c_declarationt();
}

static void init(exprt **expr, const std::string &id)
{
  init(expr);
  (*expr)->id(id);
}

static void init(typet **type, const std::string &id)
{
  init(type);
  (*type)->id(id);
}

static void set(irept &expr, const std::string &id)
{
  expr.id(id);
}

static void statement(exprt &expr, const std::string &id)
{
  set(expr, "code");
  expr.statement(id);
}

static void decl_statement(exprt &dest, ansi_c_declarationt &decl, exprt &initializer)
{
  codet decl_statement("decl");

  symbol_exprt symbol;
  symbol.set_identifier(decl.name());

  decl_statement.move_to_operands(symbol);

  if(initializer.is_not_nil())
  {
    decl.decl_value()=initializer;
    decl_statement.move_to_operands(initializer);
  }

  PARSER.move_declaration(decl);

  dest.move_to_operands(decl_statement);
}

static void merge_types(typet &dest, typet &src)
{

  if (dest.id() == "code") {
    merge_types(dest.subtype(), src);
    return;
  } else if (src.id() == "code") {
    merge_types(dest, src.subtype());
    src.subtype() = dest;
    dest = src;
    return;
  }

  if(dest.id()!="merged_type")
  {
    locationt location=dest.location();
    typet new_type("merged_type");
    new_type.move_to_subtypes((typet &)(dest));
    dest.swap(new_type);
    dest.location()=location;
  }

  ((typet &)dest).move_to_subtypes((typet &)(src));
}

static void merge_types(typet *dest, typet *src)
{

  merge_types(*dest, *src);
  return;
}

static void make_subtype(typet &dest, typet &src)
{
  // inserts "src" into "dest"
  // e.g., src is a pointer or array,
  // dest is a symbol or type

  // find spot where to insert this

  // Inserting a nil type makes precisely no change.
  if (src.is_nil())
    return;

  assert(src.is_array() ||
         src.id()=="incomplete_array" ||
         src.id()=="pointer" ||
         src.is_code() ||
         src.id()=="merged_type" ||
         src.id()=="declarator");

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
       sub->is_array() ||
       sub->id()=="incomplete_array" ||
       sub->is_code() ||
       sub->id()=="declarator")
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

static void do_pointer(typet &ptr, typet &dest)
{
  set(ptr, "pointer");
  ptr.subtype().make_nil();
  make_subtype(dest, ptr);
}

// XXX - fix type
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

    ansi_c_declarationt &decl = (ansi_c_declarationt&)*it; // Casting you say...
    exprt &v=decl.decl_value();
    if(v.is_nil())
      v=value;

    exprt symbol_expr("symbol");
    symbol_expr.identifier(it->name());

    value=exprt("+");
    value.copy_to_operands(symbol_expr, gen_one(int_type()));

    PARSER.copy_declaration(*it);
  }
}

static void do_tag(typet &key, typet &tag)
{
  irep_idt id_class_str=tag.id_class();

  ansi_c_id_classt id_class=
    (ansi_c_id_classt)atoi(id_class_str.c_str());

  if(id_class==ANSI_C_TAG) // we have it already
    return;

  // it's new
  ansi_c_declarationt declaration;
  PARSER.new_declaration(key, tag, declaration, true);

  // grab symbol
  tag.id("symbol");
  tag.identifier(declaration.get_name());
  tag.location()=declaration.location();

  declaration.type().id("incomplete_"+declaration.type().id_string());
  PARSER.move_declaration(declaration);
}

static void create_function_scope(exprt &expr)
{
  ansi_c_declarationt &declaration=to_ansi_c_declaration(expr);

  PARSER.function=declaration.get_base_name();

  std::string prefix=PARSER.current_scope().prefix+
                     id2string(declaration.get_base_name())+"::";
  PARSER.new_scope(prefix);

  if(declaration.type().is_code())
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

static void
move_decl_info_upwards(typet &dest, typet &src)
{

  if (!src.is_decl_ident_set())
    return;

  dest.decl_ident(src.decl_ident());
  src.remove("decl_ident");
  return;
}
