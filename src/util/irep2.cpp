#include <ac_config.h>
#include <boost/algorithm/string.hpp>
#include <boost/functional/hash.hpp>
#include <boost/static_assert.hpp>
#include <cstdarg>
#include <cstring>
#include <sstream>
#include <util/i2string.h>
#include <util/irep2.h>
#include <util/migrate.h>
#include <util/std_types.h>

#ifdef WITH_PYTHON
#include <boost/python.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/object/find_instance.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <util/bp_converter.h>

// Additional python infrastructure: our irep containers don't quite match
// the pointer ownership model that boost.python expects. Specifically: it
// either stores values by value, or by _boost_ shared ptrs. The former isn't
// compatible with our irep model (everything must be held by one of our own
// std::shared_ptrs), and the latter requires a large number of hoops to be
// jumped through which will only really work with boost::shared_ptr's. To
// get around this, we hack it, in what's actually a safe way.
//
// To elaborate: the boost shared ptr activity boost.python performs is to
// register a deleter method with the shared_ptr, that gets called when the
// reference count hits zero. As far as I can tell, boost.python then stores
// objects by value, but will export a boost shared_ptr to the value when
// asked. This creates a scenario where the value may be referred to by:
//  * The Python object
//  * Shared ptr's stored in some C++ code somewhere.
// Which is naturally messy.
//
// With the deleter, a python reference is kept to the python object, keeping
// it in memory, so long as shared_ptr's point at it from C++. That ensures
// that, so long as _either_ python or C++ have a ref to the object, it's kept
// alive. As a side-effect, this also means that the python object can be
// kept alive long after any python code stops running. (It might be that
// boost.python actually only stores a shared_ptr, and the object still lives
// on the heap, dunno why the deleter dance is performed in that case. Or
// perhaps it stores both).
//
// For ESBMC, we can definitively say that only containers are ever built by
// boost.python, because none of the irep constructors are exposed to it, so
// it never stores an irep by value. Because objects are always in containers,
// there's no need to worry about the lifetime of a python instance: it'll just
// decrement the shared_ptr ref count when it gets destroyed.
//
// To impose this policy upon boost.python, we register a to irep2t converter
// that sucks the corresponding container out of the python instance, and then
// just returns the irep2t pointer. There's no opportunity to enforce const
// correctness: we just shouldn't register any mutable methods.
//
// Some boost.python storage model documentation would not be amiss :/

// shared_ptr_from_python exists in the boost::python::objects:: namespace,
// but inserting our own specialization doesn't get resolved, instead the
// important part is that the converter is registered in the ::insert method.

template <typename Container, typename Base>
class irep2tc_to_irep2t
{
public:
  static void* rvalue_cvt(const Container *type, Base *out)
  {
    (void)type;
    (void)out;
    // Everything here should have become an lvalue over an rvalue. Only thing
    // that should pass through this far is None.
    std::cerr << "rvalue of irep2tc_to_irep2t should never be called" << std::endl;
    abort();
  }

  static void *lvalue_cvt(const Container *foo)
  {
    return const_cast<void *>(reinterpret_cast<const void*>((foo)->get()));
  }
};

// Facility to build one kind of container from another. For some reason, b.p
// doesn't currently want to upcast a symbol2tc to a expr2tc. It has sufficient
// information to work that out (it knows expr2t is a base of symbol2t) but
// it seems because the container types are different, it won't do it. So,
// encode that conversion manually.
template <typename DownType, typename BaseType>
class irep2tc_to_irep2tc
{
public:
  static void* rvalue_cvt(const DownType *type, BaseType *out)
  {
    // Potentially build a null irep, otherwise build a base2tc out of the
    // derived class.
    if (reinterpret_cast<const PyObject*>(type) == Py_None)
      new (out) BaseType();
    else
      new (out) BaseType(*type);

    return const_cast<void*>(reinterpret_cast<const void *>(out));
  }

  static void *lvalue_cvt(const DownType *foo)
  {
    // Cast derived type down to base type ptr
    return const_cast<void *>(reinterpret_cast<const void*>(
          dynamic_cast<const BaseType*>(foo)));
  }
};

template<typename Container>
class none_to_irep2tc
{
public:
  static void *rvalue_cvt(const char *src, Container *type)
  {
    // Everything here should have become an lvalue over an rvalue. Only thing
    // that should pass through this far is None
    assert(reinterpret_cast<const PyObject*>(src) == Py_None);
    new (type) Container(); // Empty
    (void)src; // unused
    return (void*)type;
  }

  static void *lvalue_cvt(const char *foo)
  {
    return const_cast<void *>(reinterpret_cast<const void*>((foo)));
  }
};

// Extra bonus point fun: if we're using boost python, then additional
// juggling is required to extract what the pointee type is from our shared
// pointer class
namespace boost {
  namespace python {
    template <typename T1, typename T2, unsigned int T3, typename T4, T4 T1::*T5, typename T6>
    struct pointee<esbmct::something2tc<T1, T2, T3, T4, T5, T6> > {
      typedef T2 type;
    };
  }
}

#endif

template <typename T> class register_irep_methods;

std::string
indent_str(unsigned int indent)
{
  return std::string(indent, ' ');
}

template <class T>
std::string
pretty_print_func(unsigned int indent, std::string ident, T obj)
{
  list_of_memberst memb = obj.tostring(indent+2);

  std::string indentstr = indent_str(indent);
  std::string exprstr = ident;

  for (list_of_memberst::const_iterator it = memb.begin(); it != memb.end();
       it++) {
    exprstr += "\n" + indentstr + "* " + it->first + " : " + it->second;
  }

  return exprstr;
}

/*************************** Base type2t definitions **************************/

static std::vector<std::string> illegal_python_names = {"not", "or", "and", "with"};

static const char *type_names[] = {
  "bool",
  "empty",
  "symbol",
  "struct",
  "union",
  "code",
  "array",
  "pointer",
  "unsignedbv",
  "signedbv",
  "fixedbv",
  "floatbv",
  "string",
  "cpp_name"
};
// If this fires, you've added/removed a type id, and need to update the list
// above (which is ordered according to the enum list)
BOOST_STATIC_ASSERT(sizeof(type_names) ==
                    (type2t::end_type_id * sizeof(char *)));

std::string
get_type_id(const type2t &type)
{
  return std::string(type_names[type.type_id]);
}


type2t::type2t(type_ids id)
  : type_id(id),
    crc_val(0)
{
}

type2t::type2t(const type2t &ref)
  : type_id(ref.type_id),
    crc_val(ref.crc_val)
{
}

bool
type2t::operator==(const type2t &ref) const
{

  return cmpchecked(ref);
}

bool
type2t::operator!=(const type2t &ref) const
{

  return !cmpchecked(ref);
}

bool
type2t::operator<(const type2t &ref) const
{
  int tmp = type2t::lt(ref);
  if (tmp < 0)
    return true;
  else if (tmp > 0)
    return false;
  else
    return (lt(ref) < 0);
}

int
type2t::ltchecked(const type2t &ref) const
{
  int tmp = type2t::lt(ref);
  if (tmp != 0)
    return tmp;

  return lt(ref);
}

bool
type2t::cmpchecked(const type2t &ref) const
{

  if (type_id == ref.type_id)
    return cmp(ref);

  return false;
}

int
type2t::lt(const type2t &ref) const
{

  if (type_id < ref.type_id)
    return -1;
  if (type_id > ref.type_id)
    return 1;
  return 0;
}

std::string
type2t::pretty(unsigned int indent) const
{

  return pretty_print_func<const type2t&>(indent, type_names[type_id], *this);
}

void
type2t::dump(void) const
{
  std::cout << pretty(0) << std::endl;
  return;
}

uint32_t
type2t::crc(void) const
{
  size_t seed = 0;
  do_crc(seed);
  return seed;
}

size_t
type2t::do_crc(size_t seed) const
{
  boost::hash_combine(seed, (uint8_t)type_id);
  return seed;
}

void
type2t::hash(crypto_hash &hash) const
{
  BOOST_STATIC_ASSERT(type2t::end_type_id < 256);
  uint8_t tid = type_id;
  hash.ingest(&tid, sizeof(tid));
  return;
}

unsigned int
bool_type2t::get_width(void) const
{
  // For the purpose of the byte representating memory model
  return 8;
}

unsigned int
bv_data::get_width(void) const
{
  return width;
}

unsigned int
array_type2t::get_width(void) const
{
  // Two edge cases: the array can have infinite size, or it can have a dynamic
  // size that's determined by the solver.
  if (size_is_infinite)
    throw new inf_sized_array_excp();

  if (array_size->expr_id != expr2t::constant_int_id)
    throw new dyn_sized_array_excp(array_size);

  // Otherwise, we can multiply the size of the subtype by the number of elements.
  unsigned int sub_width = subtype->get_width();

  const expr2t *elem_size = array_size.get();
  const constant_int2t *const_elem_size = dynamic_cast<const constant_int2t*>
                                                      (elem_size);
  assert(const_elem_size != NULL);
  unsigned long num_elems = const_elem_size->as_ulong();

  return num_elems * sub_width;
}

unsigned int
pointer_type2t::get_width(void) const
{
  return config.ansi_c.pointer_width;
}

unsigned int
empty_type2t::get_width(void) const
{
  throw new symbolic_type_excp();
}

unsigned int
symbol_type2t::get_width(void) const
{
  std::cerr <<"Fetching width of symbol type - invalid operation" << std::endl;
  abort();
}

unsigned int
cpp_name_type2t::get_width(void) const
{
  std::cerr << "Fetching width of cpp_name type - invalid operation" << std::endl;
  abort();
}

unsigned int
struct_type2t::get_width(void) const
{
  // Iterate over members accumulating width.
  std::vector<type2tc>::const_iterator it;
  unsigned int width = 0;
  for (it = members.begin(); it != members.end(); it++)
    width += (*it)->get_width();

  return width;
}

unsigned int
union_type2t::get_width(void) const
{
  // Iterate over members accumulating width.
  std::vector<type2tc>::const_iterator it;
  unsigned int width = 0;
  for (it = members.begin(); it != members.end(); it++)
    width = std::max(width, (*it)->get_width());

  return width;
}

unsigned int
fixedbv_type2t::get_width(void) const
{
  return width;
}

unsigned int
floatbv_type2t::get_width(void) const
{
  return fraction + exponent + 1;
}

unsigned int
code_data::get_width(void) const
{
  throw new symbolic_type_excp();
}

unsigned int
string_type2t::get_width(void) const
{
  return width * 8;
}

const std::vector<type2tc> &
struct_union_data::get_structure_members(void) const
{
  return members;
}

const std::vector<irep_idt> &
struct_union_data::get_structure_member_names(void) const
{
  return member_names;
}

const irep_idt &
struct_union_data::get_structure_name(void) const
{
  return name;
}

unsigned int
struct_union_data::get_component_number(const irep_idt &name) const
{

  unsigned int i = 0;
  forall_names(it, member_names) {
    if (*it == name)
      return i;
    i++;
  }

  std::cerr << "Looking up index of nonexistant member \"" << name
            << "\" in struct/union \"" << name << "\"" << std::endl;
  abort();
}

#ifdef WITH_PYTHON
template<>
class register_irep_methods<type2t>
{
public:
  template <typename O>
  void operator()(O &o, const std::string &irep_name __attribute__((unused)))
  {
    // Define standard methods
    o.def("pretty", &type2t::pretty);
    o.def("crc", &type2t::crc);
    o.def("clone", &type2t::clone);
    o.def_readonly("type_id", &type2t::type_id);

    // Operators. Can't use built-in boost.python way because it only
    // compares the base object, not the container. So, define our equality
    // operator to work on the containers. First resolve overload:
    bool (*eqptr)(const type2tc &a, const type2tc &b) = &operator==;
    o.def("__eq__", eqptr);
    o.def("__req__", eqptr);

    bool (*neptr)(const type2tc &a, const type2tc &b) = &operator!=;
    o.def("__ne__", neptr);
    o.def("__rne__", neptr);

    bool (*ltptr)(const type2tc &a, const type2tc &b) = &operator<;
    o.def("__lt__", ltptr);

    o.def("__hash__", &type2t::crc);

    return;
  }
};

void
build_base_type2t_python_class(void)
{
  using namespace boost::python;
  class_<type2t, boost::noncopyable, irep_container<type2t> > foo("type2t", no_init);
  register_irep_methods<type2t> bar;
  bar(foo, "type");

  // Register our manual type2tc -> type2t converter.
  esbmc_python_cvt<type2t, type2tc, false, true, true, irep2tc_to_irep2t<type2tc, type2t> >();
  // None converter
  esbmc_python_cvt<type2tc, char, true, true, false, none_to_irep2tc<type2tc> >();

  enum_<type2t::type_ids>("type_ids")
#define hahatemporary(r, data, elem) .value(BOOST_PP_STRINGIZE(elem), type2t::BOOST_PP_CAT(elem,_id))
BOOST_PP_LIST_FOR_EACH(hahatemporary, foo, ESBMC_LIST_OF_TYPES)
#undef hahatemporary
  ;

  // Should really be at top level
  class_<std::vector<type2tc> >("type_vec")
    .def(vector_indexing_suite<std::vector<type2tc> >());
}

void
build_type2t_container_converters(void)
{
  // Needs to be called _after_ the type types are registered.
#define hahatemporary(r, data, elem) \
  esbmc_python_cvt<BOOST_PP_CAT(elem,_type2t), BOOST_PP_CAT(elem,_type2tc), false, true, true, irep2tc_to_irep2t<BOOST_PP_CAT(elem,_type2tc), BOOST_PP_CAT(elem,_type2t)> >();
BOOST_PP_LIST_FOR_EACH(hahatemporary, foo, ESBMC_LIST_OF_TYPES)
#undef hahatemporary

#define hahatemporary(r, data, elem) \
  esbmc_python_cvt<type2tc, BOOST_PP_CAT(elem,_type2tc), true, true, true, irep2tc_to_irep2tc<BOOST_PP_CAT(elem,_type2tc),type2tc> >();
BOOST_PP_LIST_FOR_EACH(hahatemporary, foo, ESBMC_LIST_OF_TYPES)
#undef hahatemporary

  return;
}
#endif

namespace esbmct {
template <typename ...Args>
template <typename derived>
auto
type2t_traits<Args...>::make_contained(typename Args::result_type... args) -> irep_container<base2t> {
  return irep_container<base2t>(new derived(args...));
}
}

/*************************** Base expr2t definitions **************************/

expr2t::expr2t(const type2tc _type, expr_ids id)
  : std::enable_shared_from_this<expr2t>(), expr_id(id), type(_type), crc_val(0)
{
}

expr2t::expr2t(const expr2t &ref)
  : std::enable_shared_from_this<expr2t>(), expr_id(ref.expr_id),
    type(ref.type),
    crc_val(ref.crc_val)
{
}

bool
expr2t::operator==(const expr2t &ref) const
{
  if (!expr2t::cmp(ref))
    return false;

  return cmp(ref);
}

bool
expr2t::operator!=(const expr2t &ref) const
{
  return !(*this == ref);
}

bool
expr2t::operator<(const expr2t &ref) const
{
  int tmp = expr2t::lt(ref);
  if (tmp < 0)
    return true;
  else if (tmp > 0)
    return false;
  else
    return (lt(ref) < 0);
}

unsigned long
expr2t::depth(void) const
{
  unsigned long num_nodes = 0;

  for (unsigned int idx = 0; idx < get_num_sub_exprs(); idx++) {
    const expr2tc *e = get_sub_expr(idx);
    if (is_nil_expr(*e))
      continue;
    unsigned long tmp = (*e)->depth();
    num_nodes = std::max(num_nodes, tmp);
  }

  num_nodes++; // Count ourselves.
  return num_nodes;
}

unsigned long
expr2t::num_nodes(void) const
{
  unsigned long count = 0;

  for (unsigned int idx = 0; idx < get_num_sub_exprs(); idx++) {
    const expr2tc *e = get_sub_expr(idx);
    if (is_nil_expr(*e))
      continue;
    count += (*e)->num_nodes();
  }

  count++; // Count ourselves.
  return count;
}

int
expr2t::ltchecked(const expr2t &ref) const
{
  int tmp = expr2t::lt(ref);
  if (tmp != 0)
    return tmp;

  return lt(ref);
}

bool
expr2t::cmp(const expr2t &ref) const
{
  if (expr_id != ref.expr_id)
    return false;

  if (type != ref.type)
    return false;

  return true;
}

int
expr2t::lt(const expr2t &ref) const
{
  if (expr_id < ref.expr_id)
    return -1;
  if (expr_id > ref.expr_id)
    return 1;

  return type->ltchecked(*ref.type.get());
}

uint32_t
expr2t::crc(void) const
{
  size_t seed = 0;
  return do_crc(seed);
}

size_t
expr2t::do_crc(size_t seed) const
{
  boost::hash_combine(seed, (uint8_t)expr_id);
  return type->do_crc(seed);
}

void
expr2t::hash(crypto_hash &hash) const
{
  BOOST_STATIC_ASSERT(expr2t::end_expr_id < 256);
  uint8_t eid = expr_id;
  hash.ingest(&eid, sizeof(eid));
  type->hash(hash);
  return;
}

expr2tc
expr2t::simplify(void) const
{
  try {

  // Corner case! Don't even try to simplify address of's operands, might end up
  // taking the address of some /completely/ arbitary pice of data, by
  // simplifiying an index to its data, discarding the symbol.
  if (__builtin_expect((expr_id == address_of_id), 0)) // unlikely
    return expr2tc();

  // And overflows too. We don't wish an add to distribute itself, for example,
  // when we're trying to work out whether or not it's going to overflow.
  if (__builtin_expect((expr_id == overflow_id), 0))
    return expr2tc();

  // Try initial simplification
  expr2tc res = do_simplify();
  if (!is_nil_expr(res)) {
    // Woot, we simplified some of this. It may have _additional_ fields that
    // need to get simplified (member2ts in arrays for example), so invoke the
    // simplifier again, to hit those potential subfields.
    expr2tc res2 = res->simplify();

    // If we simplified even further, return res2; otherwise res.
    if (is_nil_expr(res2))
      return res;
    else
      return res2;
  }

  // Try simplifying all the sub-operands.
  bool changed = false;
  std::list<expr2tc> newoperands;

  for (unsigned int idx = 0; idx < get_num_sub_exprs(); idx++) {
    const expr2tc *e = get_sub_expr(idx);
    expr2tc tmp;

    if (!is_nil_expr(*e)) {
      tmp = e->get()->simplify();
      if (!is_nil_expr(tmp))
        changed = true;
    }

    newoperands.push_back(tmp);
  }

  if (changed == false)
    // Second shot at simplification. For efficiency, a simplifier may be
    // holding something back until it's certain all its operands are
    // simplified. It's responsible for simplifying further if it's made that
    // call though.
    return do_simplify(true);

  // An operand has been changed; clone ourselves and update.
  expr2tc new_us = clone();
  std::list<expr2tc>::iterator it2 = newoperands.begin();
  new_us.get()->Foreach_operand([this, &it2] (expr2tc &e) {
      if ((*it2) == NULL)
        ; // No change in operand;
      else
        e = *it2; // Operand changed; overwrite with new one.
      it2++;
    }
  );

  // Finally, attempt simplification again.
  expr2tc tmp = new_us->do_simplify(true);
  if (is_nil_expr(tmp))
    return new_us;
  else
    return tmp;

  } catch (array_type2t::dyn_sized_array_excp *e) {
    // Pretty much anything in any expression could be fouled up by there
    // being a dynamically sized array somewhere in there. In this circumstance,
    // don't even attempt partial simpilfication. We'd probably have to double
    // the size of simplification code in that case.
    return expr2tc();
  }
}

static const char *expr_names[] = {
  "constant_int",
  "constant_fixedbv",
  "constant_floatbv",
  "constant_bool",
  "constant_string",
  "constant_struct",
  "constant_union",
  "constant_array",
  "constant_array_of",
  "symbol",
  "typecast",
  "bitcast",
  "nearbyint",
  "if",
  "equality",
  "notequal",
  "lessthan",
  "greaterthan",
  "lessthanequal",
  "greaterthanequal",
  "not",
  "and",
  "or",
  "xor",
  "implies",
  "bitand",
  "bitor",
  "bitxor",
  "bitnand",
  "bitnor",
  "bitnxor",
  "bitnot",
  "lshr",
  "neg",
  "abs",
  "add",
  "sub",
  "mul",
  "div",
  "ieee_add",
  "ieee_sub",
  "ieee_mul",
  "ieee_div",
  "ieee_fma",
  "modulus",
  "shl",
  "ashr",
  "dynamic_object",
  "same_object",
  "pointer_offset",
  "pointer_object",
  "address_of",
  "byte_extract",
  "byte_update",
  "with",
  "member",
  "index",
  "isnan",
  "overflow",
  "overflow_cast",
  "overflow_neg",
  "unknown",
  "invalid",
  "NULL-object",
  "dereference",
  "valid_object",
  "deallocated_obj",
  "dynamic_size",
  "sideeffect",
  "code_block",
  "code_assign",
  "code_init",
  "code_decl",
  "code_printf",
  "code_expression",
  "code_return",
  "code_skip",
  "code_free",
  "code_goto",
  "object_descriptor",
  "code_function_call",
  "code_comma_id",
  "invalid_pointer",
  "code_asm",
  "cpp_del_array",
  "cpp_delete",
  "cpp_catch",
  "cpp_throw",
  "cpp_throw_decl",
  "cpp_throw_decl_end",
  "isinf",
  "isnormal",
  "isfinite",
  "signbit",
  "concat",
};
// If this fires, you've added/removed an expr id, and need to update the list
// above (which is ordered according to the enum list)
BOOST_STATIC_ASSERT(sizeof(expr_names) ==
                    (expr2t::end_expr_id * sizeof(char *)));

std::string
get_expr_id(const expr2t &expr)
{
  return std::string(expr_names[expr.expr_id]);
}

std::string
expr2t::pretty(unsigned int indent) const
{

  std::string ret = pretty_print_func<const expr2t&>(indent,
                                                     expr_names[expr_id],
                                                     *this);
  // Dump the type on the end.
  ret += std::string("\n") + indent_str(indent) + "* type : "
         + type->pretty(indent+2);
  return ret;
}

void
expr2t::dump(void) const
{
  std::cout << pretty(0) << std::endl;
  return;
}

// Map a base type to it's list of names
template <typename T>
class base_to_names;

template<>
class base_to_names<type2t> {
public:
  static constexpr const char **names = type_names;
};

template<>
class base_to_names<expr2t> {
public:
  static constexpr const char **names = expr_names;
};

#ifdef WITH_PYTHON
template<>
class register_irep_methods<expr2t>
{
public:
  template <typename O>
  void operator()(O &o, const std::string &irep_name)
  {
    // Define standard methods
    o.def("clone", &expr2t::clone);
    o.def("pretty", &expr2t::pretty);
    o.def("num_nodes", &expr2t::num_nodes);
    o.def("depth", &expr2t::depth);
    o.def("crc", &expr2t::crc);
    o.def("simplify", &expr2t::simplify);
    o.def_readonly("type", &expr2t::type);
    o.def_readonly("expr_id", &expr2t::expr_id);

    // Operators. Can't use built-in boost.python way because it only
    // compares the base object, not the container. So, define our equality
    // operator to work on the containers, and refs. First resolve overload:
    bool (*eqptr)(const expr2tc &a, const expr2tc &b) = &operator==;
    o.def("__eq__", eqptr);
    o.def("__req__", eqptr);

    bool (*neptr)(const expr2tc &a, const expr2tc &b) = &operator!=;
    o.def("__ne__", neptr);
    o.def("__rne__", neptr);

    bool (*ltptr)(const expr2tc &a, const expr2tc &b) = &operator<;
    o.def("__lt__", ltptr);

    o.def("__hash__", &expr2t::crc);

    // Register super special irep methods
    if (irep_name == "symbol")
      o.def("get_symbol_name", &symbol2t::get_symbol_name);
    return;
  }
};

template <typename T>
void
filter_illegal_python_names(T enumids, const char *str, expr2t::expr_ids id)
{
  std::string filtered_pyname = std::string(str);

  if (std::find(illegal_python_names.begin(), illegal_python_names.end(), filtered_pyname) != illegal_python_names.end())
    filtered_pyname.append("_");

  enumids.value(filtered_pyname.c_str(), id);
  return;
}

void
build_base_expr2t_python_class(void)
{
  using namespace boost::python;
  class_<expr2t, boost::noncopyable, irep_container<expr2t> > foo("expr2t", no_init);
  register_irep_methods<expr2t> bar;
  bar(foo, "expr2t");

  // Register our manual expr2tc -> expr2t converter.
  esbmc_python_cvt<expr2t, expr2tc, false, true, true, irep2tc_to_irep2t<expr2tc, expr2t> >();
  // None converter
  esbmc_python_cvt<expr2tc, char, true, true, false, none_to_irep2tc<expr2tc> >();

  {
  enum_<expr2t::expr_ids> enumids("expr_ids");
#define hahatemporary(r, data, elem) filter_illegal_python_names(enumids, BOOST_PP_STRINGIZE(elem), expr2t::BOOST_PP_CAT(elem,_id));
BOOST_PP_LIST_FOR_EACH(hahatemporary, foo, ESBMC_LIST_OF_EXPRS)
#undef hahatemporary
  }

  // Register some additional enumerations. These should be inside the relevant
  // expr classes... but I don't think we can get a handle on the class_
  // object for that.
  enum_<symbol_data::renaming_level>("symbol_renaming")
    .value("level0", symbol_data::renaming_level::level0)
    .value("level1", symbol_data::renaming_level::level1)
    .value("level2", symbol_data::renaming_level::level2)
    .value("level1_global", symbol_data::renaming_level::level1_global)
    .value("level2_global", symbol_data::renaming_level::level2_global);

  enum_<sideeffect_data::allockind>("sideeffect_allockind")
    .value("malloc", sideeffect_data::allockind::malloc)
    .value("alloca", sideeffect_data::allockind::alloca)
    .value("cpp_new", sideeffect_data::allockind::cpp_new)
    .value("cpp_new_arr", sideeffect_data::allockind::cpp_new_arr)
    .value("nondet", sideeffect_data::allockind::nondet)
    .value("function_call", sideeffect_data::allockind::function_call);

  // We can use boost magic to define what a vector looks like!
  // Should really be at top level
  class_<std::vector<expr2tc> >("expr_vec")
    .def(vector_indexing_suite<std::vector<expr2tc> >());
}

void
build_expr2t_container_converters(void)
{
  // Needs to be called _after_ the expr types are registered.
#define hahatemporary(r, data, elem) \
  esbmc_python_cvt<BOOST_PP_CAT(elem,2t), BOOST_PP_CAT(elem,2tc), false, true, true, irep2tc_to_irep2t<BOOST_PP_CAT(elem,2tc), BOOST_PP_CAT(elem,2t)> >();
BOOST_PP_LIST_FOR_EACH(hahatemporary, foo, ESBMC_LIST_OF_EXPRS)
#undef hahatemporary

#define hahatemporary(r, data, elem) \
  esbmc_python_cvt<expr2tc, BOOST_PP_CAT(elem,2tc), true, true, true, irep2tc_to_irep2tc<BOOST_PP_CAT(elem,2tc),expr2tc> >();
BOOST_PP_LIST_FOR_EACH(hahatemporary, foo, ESBMC_LIST_OF_EXPRS)
#undef hahatemporary

  return;
}
#endif

// Undoubtedly a better way of doing this...
namespace esbmct {
template <typename ...Args>
template <typename derived>
auto
expr2t_traits<Args...>::make_contained(const type2tc &type, typename Args::result_type... args) -> irep_container<base2t> {
  return irep_container<base2t>(new derived(type, args...));
}

template <typename ...Args>
template <typename derived>
auto
expr2t_traits_notype<Args...>::make_contained(typename Args::result_type... args) -> irep_container<base2t> {
  return irep_container<base2t>(new derived(args...));
}

template <typename ...Args>
template <typename derived>
auto
expr2t_traits_always_construct<Args...>::make_contained(typename Args::result_type... args) -> irep_container<base2t> {
  return irep_container<base2t>(new derived(args...));
}
}

/**************************** Expression constructors *************************/

unsigned long
constant_int2t::as_ulong(void) const
{
  // XXXjmorse - add assertion that we don't exceed machine word width?
  assert(!value.is_negative());
  return value.to_ulong();
}

long
constant_int2t::as_long(void) const
{
  // XXXjmorse - add assertion that we don't exceed machine word width?
  return value.to_long();
}

bool
constant_bool2t::is_true(void) const
{
  return value;
}

bool
constant_bool2t::is_false(void) const
{
  return !value;
}

std::string
symbol_data::get_symbol_name(void) const
{
  switch (rlevel) {
  case level0:
    return thename.as_string();
  case level1:
    return thename.as_string() + "@" + i2string(level1_num)
                               + "!" + i2string(thread_num);
  case level2:
    return thename.as_string() + "@" + i2string(level1_num)
                               + "!" + i2string(thread_num)
                               + "&" + i2string(node_num)
                               + "#" + i2string(level2_num);
  case level1_global:
    // Just return global name,
    return thename.as_string();
  case level2_global:
    // Global name with l2 details
    return thename.as_string() + "&" + i2string(node_num)
                               + "#" + i2string(level2_num);
  default:
    std::cerr << "Unrecognized renaming level enum" << std::endl;
    abort();
  }
}

expr2tc
constant_string2t::to_array(void) const
{
  std::vector<expr2tc> contents;
  unsigned int length = value.as_string().size(), i;

  type2tc type = type_pool.get_uint8();

  for (i = 0; i < length; i++) {
    constant_int2t *v = new constant_int2t(type, BigInt(value.as_string()[i]));
    expr2tc ptr(v);
    contents.push_back(ptr);
  }

  // Null terminator is implied.
  contents.push_back(expr2tc(new constant_int2t(type, BigInt(0))));

  unsignedbv_type2t *len_type = new unsignedbv_type2t(config.ansi_c.int_width);
  type2tc len_tp(len_type);
  constant_int2t *len_val = new constant_int2t(len_tp, BigInt(contents.size()));
  expr2tc len_val_ref(len_val);

  array_type2t *arr_type = new array_type2t(type, len_val_ref, false);
  type2tc arr_tp(arr_type);
  constant_array2t *a = new constant_array2t(arr_tp, contents);

  expr2tc final_val(a);
  return final_val;
}

const expr2tc &
object_descriptor2t::get_root_object(void) const
{
  const expr2tc *tmp = &object;

  do {
    if (is_member2t(*tmp))
      tmp = &to_member2t(*tmp).source_value;
    else if (is_index2t(*tmp))
      tmp = &to_index2t(*tmp).source_value;
    else
      return *tmp;
  } while (1);
}

type_poolt::type_poolt(void)
{
  // This space is deliberately left blank
}

type_poolt::type_poolt(bool yolo __attribute__((unused)))
{
  bool_type = type2tc(new bool_type2t());
  empty_type = type2tc(new empty_type2t());

  // Create some int types.
  type2tc ubv8(new unsignedbv_type2t(8));
  type2tc ubv16(new unsignedbv_type2t(16));
  type2tc ubv32(new unsignedbv_type2t(32));
  type2tc ubv64(new unsignedbv_type2t(64));
  type2tc sbv8(new signedbv_type2t(8));
  type2tc sbv16(new signedbv_type2t(16));
  type2tc sbv32(new signedbv_type2t(32));
  type2tc sbv64(new signedbv_type2t(64));

  unsignedbv_map[unsignedbv_typet(8)] = ubv8;
  unsignedbv_map[unsignedbv_typet(16)] = ubv16;
  unsignedbv_map[unsignedbv_typet(32)] = ubv32;
  unsignedbv_map[unsignedbv_typet(64)] = ubv64;
  signedbv_map[signedbv_typet(8)] = sbv8;
  signedbv_map[signedbv_typet(16)] = sbv16;
  signedbv_map[signedbv_typet(32)] = sbv32;
  signedbv_map[signedbv_typet(64)] = sbv64;

  uint8 = &unsignedbv_map[unsignedbv_typet(8)];
  uint16 = &unsignedbv_map[unsignedbv_typet(16)];
  uint32 = &unsignedbv_map[unsignedbv_typet(32)];
  uint64 = &unsignedbv_map[unsignedbv_typet(64)];
  int8 = &signedbv_map[signedbv_typet(8)];
  int16 = &signedbv_map[signedbv_typet(16)];
  int32 = &signedbv_map[signedbv_typet(32)];
  int64 = &signedbv_map[signedbv_typet(64)];

  return;
}

type_poolt &
type_poolt::operator=(type_poolt const &ref)
{
  bool_type = ref.bool_type;
  empty_type = ref.empty_type;
  struct_map = ref.struct_map;
  union_map = ref.union_map;
  array_map = ref.array_map;
  pointer_map = ref.pointer_map;
  unsignedbv_map = ref.unsignedbv_map;
  signedbv_map = ref.signedbv_map;
  fixedbv_map = ref.fixedbv_map;
  floatbv_map = ref.floatbv_map;
  string_map = ref.string_map;
  code_map = ref.code_map;

  // Re-establish some pointers
  uint8 = &unsignedbv_map[unsignedbv_typet(8)];
  uint16 = &unsignedbv_map[unsignedbv_typet(16)];
  uint32 = &unsignedbv_map[unsignedbv_typet(32)];
  uint64 = &unsignedbv_map[unsignedbv_typet(64)];
  int8 = &signedbv_map[signedbv_typet(8)];
  int16 = &signedbv_map[signedbv_typet(16)];
  int32 = &signedbv_map[signedbv_typet(32)];
  int64 = &signedbv_map[signedbv_typet(64)];

  return *this;
}

// XXX investigate performance implications of this cache
static const type2tc &
get_type_from_pool(const typet &val,
    std::map<typet, type2tc> &map __attribute__((unused)))
{
#if 0
  std::map<const typet, type2tc>::const_iterator it = map.find(val);
  if (it != map.end())
    return it->second;
#endif

  type2tc new_type;
  real_migrate_type(val, new_type);
#if 0
  map[val] = new_type;
  return map[val];
#endif
  return *(new type2tc(new_type));
}

const type2tc &
type_poolt::get_struct(const typet &val)
{
  return get_type_from_pool(val, struct_map);
}

const type2tc &
type_poolt::get_union(const typet &val)
{
  return get_type_from_pool(val, union_map);
}

const type2tc &
type_poolt::get_array(const typet &val)
{
  return get_type_from_pool(val, array_map);
}

const type2tc &
type_poolt::get_pointer(const typet &val)
{
  return get_type_from_pool(val, pointer_map);
}

const type2tc &
type_poolt::get_unsignedbv(const typet &val)
{
  return get_type_from_pool(val, unsignedbv_map);
}

const type2tc &
type_poolt::get_signedbv(const typet &val)
{
  return get_type_from_pool(val, signedbv_map);
}

const type2tc &
type_poolt::get_fixedbv(const typet &val)
{
  return get_type_from_pool(val, fixedbv_map);
}

const type2tc &
type_poolt::get_floatbv(const typet &val)
{
  return get_type_from_pool(val, floatbv_map);
}

const type2tc &
type_poolt::get_string(const typet &val)
{
  return get_type_from_pool(val, string_map);
}

const type2tc &
type_poolt::get_symbol(const typet &val)
{
  return get_type_from_pool(val, symbol_map);
}

const type2tc &
type_poolt::get_code(const typet &val)
{
  return get_type_from_pool(val, code_map);
}

const type2tc &
type_poolt::get_uint(unsigned int size)
{
  switch (size) {
  case 8:
    return get_uint8();
  case 16:
    return get_uint16();
  case 32:
    return get_uint32();
  case 64:
    return get_uint64();
  default:
    return get_unsignedbv(unsignedbv_typet(size));
  }
}

const type2tc &
type_poolt::get_int(unsigned int size)
{
  switch (size) {
  case 8:
    return get_int8();
  case 16:
    return get_int16();
  case 32:
    return get_int32();
  case 64:
    return get_int64();
  default:
    return get_signedbv(signedbv_typet(size));
  }
}

type_poolt type_pool;

// For CRCing to actually be accurate, expr/type ids mustn't overflow out of
// a byte. If this happens then a) there are too many exprs, and b) the expr
// crcing code has to change.
BOOST_STATIC_ASSERT(type2t::end_type_id <= 256);
BOOST_STATIC_ASSERT(expr2t::end_expr_id <= 256);

static inline __attribute__((always_inline)) std::string
type_to_string(const bool &thebool, int indent __attribute__((unused)))
{
  return (thebool) ? "true" : "false";
}

static inline __attribute__((always_inline)) std::string
type_to_string(const sideeffect_data::allockind &data,
               int indent __attribute__((unused)))
{
  return (data == sideeffect_data::allockind::malloc) ? "malloc" :
         (data == sideeffect_data::allockind::realloc) ? "realloc" :
         (data == sideeffect_data::allockind::alloca) ? "alloca" :
         (data == sideeffect_data::allockind::cpp_new) ? "cpp_new" :
         (data == sideeffect_data::allockind::cpp_new_arr) ? "cpp_new_arr" :
         (data == sideeffect_data::allockind::nondet) ? "nondet" :
         (data == sideeffect_data::allockind::va_arg) ? "va_arg" :
         (data == sideeffect_data::allockind::function_call) ? "function_call" :
         "unknown";
}

static inline __attribute__((always_inline)) std::string
type_to_string(const unsigned int &theval, int indent __attribute__((unused)))
{
  char buffer[64];
  snprintf(buffer, 63, "%d", theval);
  return std::string(buffer);
}

static inline __attribute__((always_inline)) std::string
type_to_string(const symbol_data::renaming_level &theval,
               int indent __attribute__((unused)))
{
  switch (theval) {
  case symbol_data::level0:
    return "Level 0";
  case symbol_data::level1:
    return "Level 1";
  case symbol_data::level2:
    return "Level 2";
  case symbol_data::level1_global:
    return "Level 1 (global)";
  case symbol_data::level2_global:
    return "Level 2 (global)";
  default:
    std::cerr << "Unrecognized renaming level enum" << std::endl;
    abort();
  }
}

static inline __attribute__((always_inline)) std::string
type_to_string(const BigInt &theint, int indent __attribute__((unused)))
{
  char buffer[256], *buf;

  buf = theint.as_string(buffer, 256);
  return std::string(buf);
}

static inline __attribute__((always_inline)) std::string
type_to_string(const fixedbvt &theval, int indent __attribute__((unused)))
{
  return theval.to_ansi_c_string();
}

static inline __attribute__((always_inline)) std::string
type_to_string(const ieee_floatt &theval, int indent __attribute__((unused)))
{
  return theval.to_ansi_c_string();
}

static inline __attribute__((always_inline)) std::string
type_to_string(const std::vector<expr2tc> &theval, int indent)
{
  char buffer[64];
  std::string astring = "\n";
  int i;

  i = 0;
  forall_exprs(it, theval) {
    snprintf(buffer, 63, "%d", i);
    buffer[63] = '\0';
    astring += indent_str(indent) + std::string(buffer) + ": " + (*it)->pretty(indent + 2) + "\n";
    i++;
  }

  return astring;
}

static inline __attribute__((always_inline)) std::string
type_to_string(const std::vector<type2tc> &theval, int indent)
{
  char buffer[64];
  std::string astring = "\n";
  int i;

  i = 0;
  forall_types(it, theval) {
    snprintf(buffer, 63, "%d", i);
    buffer[63] = '\0';
    astring += indent_str(indent) + std::string(buffer) + ": " + (*it)->pretty(indent + 2) + "\n";
    i++;
  }

  return astring;
}

static inline __attribute__((always_inline)) std::string
type_to_string(const std::vector<irep_idt> &theval,
               int indent __attribute__((unused)))
{
  char buffer[64];
  std::string astring = "\n";
  int i;

  i = 0;
  forall_names(it, theval) {
    snprintf(buffer, 63, "%d", i);
    buffer[63] = '\0';
    astring += indent_str(indent) + std::string(buffer) + ": " + (*it).as_string() + "\n";
    i++;
  }

  return astring;
}

static inline __attribute__((always_inline)) std::string
type_to_string(const expr2tc &theval, int indent)
{

  if (theval.get() != NULL)
   return theval->pretty(indent + 2);
  return "";
}

static inline __attribute__((always_inline)) std::string
type_to_string(const type2tc &theval, int indent)
{

  if (theval.get() != NULL)
    return theval->pretty(indent + 2);
  else
    return "";
}

static inline __attribute__((always_inline)) std::string
type_to_string(const irep_idt &theval, int indent __attribute__((unused)))
{
  return theval.as_string();
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const bool &side1, const bool &side2)
{
  return (side1 == side2) ? true : false;
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const unsigned int &side1, const unsigned int &side2)
{
  return (side1 == side2) ? true : false;
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const sideeffect_data::allockind &side1,
            const sideeffect_data::allockind &side2)
{
  return (side1 == side2) ? true : false;
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const symbol_data::renaming_level &side1,
            const symbol_data::renaming_level &side2)
{
  return (side1 == side2) ? true : false;
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const BigInt &side1, const BigInt &side2)
{
  // BigInt has its own equality operator.
  return (side1 == side2) ? true : false;
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const fixedbvt &side1, const fixedbvt &side2)
{
  return (side1 == side2) ? true : false;
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const ieee_floatt &side1, const ieee_floatt &side2)
{
  return (side1 == side2) ? true : false;
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const std::vector<expr2tc> &side1,
            const std::vector<expr2tc> &side2)
{
  return (side1 == side2);
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const std::vector<type2tc> &side1,
            const std::vector<type2tc> &side2)
{
  return (side1 == side2);
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const std::vector<irep_idt> &side1,
            const std::vector<irep_idt> &side2)
{
  return (side1 == side2);
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const expr2tc &side1, const expr2tc &side2)
{
  if (side1.get() == side2.get())
    return true; // Catch null
  else if (side1.get() == NULL || side2.get() == NULL)
    return false;
  else
    return (side1 == side2);
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const type2tc &side1, const type2tc &side2)
{
  if (side1.get() == side2.get())
    return true; // both null ptr check
  if (side1.get() == NULL || side2.get() == NULL)
    return false; // One of them is null, the other isn't
  return (side1 == side2);
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const irep_idt &side1, const irep_idt &side2)
{
  return (side1 == side2);
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const type2t::type_ids &id __attribute__((unused)),
            const type2t::type_ids &id2 __attribute__((unused)))
{
  return true; // Dummy field comparison.
}

static inline __attribute__((always_inline)) bool
do_type_cmp(const expr2t::expr_ids &id __attribute__((unused)),
            const expr2t::expr_ids &id2 __attribute__((unused)))
{
  return true; // Dummy field comparison.
}

static inline __attribute__((always_inline)) int
do_type_lt(const bool &side1, const bool &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  else
    return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const unsigned int &side1, const unsigned int &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  else
    return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const sideeffect_data::allockind &side1,
           const sideeffect_data::allockind &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  else
    return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const symbol_data::renaming_level &side1,
           const symbol_data::renaming_level &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  else
    return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const BigInt &side1, const BigInt &side2)
{
  // BigInt also has its own less than comparator.
  return side1.compare(side2);
}

static inline __attribute__((always_inline)) int
do_type_lt(const fixedbvt &side1, const fixedbvt &side2)
{
  if (side1 < side2)
    return -1;
  else if (side1 > side2)
    return 1;
  return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const ieee_floatt &side1, const ieee_floatt &side2)
{
  if (side1 < side2)
    return -1;
  else if (side1 > side2)
    return 1;
  return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const std::vector<expr2tc> &side1, const std::vector<expr2tc> &side2)
{


  int tmp = 0;
  std::vector<expr2tc>::const_iterator it2 = side2.begin();
  forall_exprs(it, side1) {
    tmp = (*it)->ltchecked(**it2);
    if (tmp != 0)
      return tmp;
    it2++;
  }
  return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const std::vector<type2tc> &side1, const std::vector<type2tc> &side2)
{

  if (side1.size() < side2.size())
    return -1;
  else if (side1.size() > side2.size())
    return 1;

  int tmp = 0;
  std::vector<type2tc>::const_iterator it2 = side2.begin();
  forall_types(it, side1) {
    tmp = (*it)->ltchecked(**it2);
    if (tmp != 0)
      return tmp;
    it2++;
  }
  return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const std::vector<irep_idt> &side1,
           const std::vector<irep_idt> &side2)
{
  if (side1 < side2)
    return -1;
  else if (side2 < side1)
    return 1;
  return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const expr2tc &side1, const expr2tc &side2)
{
  if (side1.get() == side2.get())
    return 0; // Catch nulls
  else if (side1.get() == NULL)
    return -1;
  else if (side2.get() == NULL)
    return 1;
  else
    return side1->ltchecked(*side2.get());
}

static inline __attribute__((always_inline)) int
do_type_lt(const type2tc &side1, const type2tc &side2)
{
  if (*side1.get() == *side2.get())
    return 0; // Both may be null;
  else if (side1.get() == NULL)
    return -1;
  else if (side2.get() == NULL)
    return 1;
  else
    return side1->ltchecked(*side2.get());
}

static inline __attribute__((always_inline)) int
do_type_lt(const irep_idt &side1, const irep_idt &side2)
{
  if (side1 < side2)
    return -1;
  if (side2 < side1)
    return 1;
  return 0;
}

static inline __attribute__((always_inline)) int
do_type_lt(const type2t::type_ids &id __attribute__((unused)),
           const type2t::type_ids &id2 __attribute__((unused)))
{
  return 0; // Dummy field comparison
}

static inline __attribute__((always_inline)) int
do_type_lt(const expr2t::expr_ids &id __attribute__((unused)),
           const expr2t::expr_ids &id2 __attribute__((unused)))
{
  return 0; // Dummy field comparison
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const bool &thebool, size_t seed)
{

  boost::hash_combine(seed, thebool);
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const bool &thebool, crypto_hash &hash)
{

  if (thebool) {
    uint8_t tval = 1;
    hash.ingest(&tval, sizeof(tval));
  } else {
    uint8_t tval = 0;
    hash.ingest(&tval, sizeof(tval));
  }
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const unsigned int &theval, size_t seed)
{

  boost::hash_combine(seed, theval);
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const unsigned int &theval, crypto_hash &hash)
{

  hash.ingest((void*)&theval, sizeof(theval));
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const sideeffect_data::allockind &theval, size_t seed)
{

  boost::hash_combine(seed, (uint8_t)theval);
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const sideeffect_data::allockind &theval, crypto_hash &hash)
{

  hash.ingest((void*)&theval, sizeof(theval));
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const symbol_data::renaming_level &theval, size_t seed)
{

  boost::hash_combine(seed, (uint8_t)theval);
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const symbol_data::renaming_level &theval, crypto_hash &hash)
{

  hash.ingest((void*)&theval, sizeof(theval));
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const BigInt &theint, size_t seed)
{
  unsigned char buffer[256];

  if (theint.dump(buffer, sizeof(buffer))) {
    // Zero has no data in bigints.
    if (theint.is_zero()) {
      boost::hash_combine(seed, 0);
    } else {
      unsigned int thelen = theint.get_len();
      thelen *= 4; // words -> bytes
      unsigned int start = 256 - thelen;
      for (unsigned int i = 0; i < thelen; i++)
        boost::hash_combine(seed, buffer[start + i]);
    }
  } else {
    // bigint is too large to fit in that static buffer. This is insane; but
    // rather than wasting time heap allocing we'll just skip recording data,
    // at the price of possible crc collisions.
    ;
  }
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const BigInt &theint, crypto_hash &hash)
{
  unsigned char buffer[256];

  if (theint.dump(buffer, sizeof(buffer))) {
    // Zero has no data in bigints.
    if (theint.is_zero()) {
      uint8_t val = 0;
      hash.ingest(&val, sizeof(val));
    } else {
      hash.ingest(buffer, theint.get_len());
    }
  } else {
    // bigint is too large to fit in that static buffer. This is insane; but
    // rather than wasting time heap allocing we'll just skip recording data,
    // at the price of possible crc collisions.
    ;
  }
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const fixedbvt &theval, size_t seed)
{

  return do_type_crc(theval.get_value(), seed);
}

static inline __attribute__((always_inline)) void
do_type_hash(const fixedbvt &theval, crypto_hash &hash)
{

  do_type_hash(theval.to_integer(), hash);
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const ieee_floatt &theval, size_t seed)
{
  // TODO: Check if this is correct
  return do_type_crc(theval.to_integer(), seed);
}

static inline __attribute__((always_inline)) void
do_type_hash(const ieee_floatt &theval, crypto_hash &hash)
{

  do_type_hash(theval.to_integer(), hash);
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const std::vector<expr2tc> &theval, size_t seed)
{
  forall_exprs(it, theval)
    (*it)->do_crc(seed);

  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const std::vector<expr2tc> &theval, crypto_hash &hash)
{
  forall_exprs(it, theval)
    (*it)->hash(hash);
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const std::vector<type2tc> &theval, size_t seed)
{
  forall_types(it, theval)
    (*it)->do_crc(seed);

  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const std::vector<type2tc> &theval, crypto_hash &hash)
{
  forall_types(it, theval)
    (*it)->hash(hash);
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const std::vector<irep_idt> &theval, size_t seed)
{
  forall_names(it, theval)
    boost::hash_combine(seed, (*it).as_string());
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const std::vector<irep_idt> &theval, crypto_hash &hash)
{
  forall_names(it, theval)
    hash.ingest((void*)(*it).as_string().c_str(), (*it).as_string().size());
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const expr2tc &theval, size_t seed)
{

  if (theval.get() != NULL)
    return theval->do_crc(seed);
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const expr2tc &theval, crypto_hash &hash)
{

  if (theval.get() != NULL)
    theval->hash(hash);
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const type2tc &theval, size_t seed)
{

  if (theval.get() != NULL)
    return theval->do_crc(seed);
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const type2tc &theval, crypto_hash &hash)
{

  if (theval.get() != NULL)
    theval->hash(hash);
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const irep_idt &theval, size_t seed)
{

  boost::hash_combine(seed, theval.as_string());
  return seed;
}

static inline __attribute__((always_inline)) void
do_type_hash(const irep_idt &theval, crypto_hash &hash)
{

  hash.ingest((void*)theval.as_string().c_str(), theval.as_string().size());
  return;
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const type2t::type_ids &i __attribute__((unused)), size_t seed)
{
  return seed; // Dummy field crc
}

static inline __attribute__((always_inline)) void
do_type_hash(const type2t::type_ids &i __attribute__((unused)),
             crypto_hash &hash __attribute__((unused)))
{
  return; // Dummy field crc
}

static inline __attribute__((always_inline)) size_t
do_type_crc(const expr2t::expr_ids &i __attribute__((unused)), size_t seed)
{
  return seed; // Dummy field crc
}

static inline __attribute__((always_inline)) void
do_type_hash(const expr2t::expr_ids &i __attribute__((unused)),
             crypto_hash &hash __attribute__((unused)))
{
  return; // Dummy field crc
}

template <typename T>
void
do_type2string(const T &thething, unsigned int idx,
               std::string (&names)[esbmct::num_type_fields],
               list_of_memberst &vec, unsigned int indent)
{
  vec.push_back(member_entryt(names[idx], type_to_string(thething, indent)));
}

template <>
void
do_type2string<type2t::type_ids>(
    const type2t::type_ids &thething __attribute__((unused)),
    unsigned int idx __attribute__((unused)),
    std::string (&names)[esbmct::num_type_fields] __attribute__((unused)),
    list_of_memberst &vec __attribute__((unused)),
    unsigned int indent __attribute__((unused)))
{
  // Do nothing; this is a dummy member.
}

template <>
void
do_type2string<const expr2t::expr_ids>(
    const expr2t::expr_ids &thething __attribute__((unused)),
    unsigned int idx __attribute__((unused)),
    std::string (&names)[esbmct::num_type_fields] __attribute__((unused)),
    list_of_memberst &vec __attribute__((unused)),
    unsigned int indent __attribute__((unused)))
{
  // Do nothing; this is a dummy member.
}

template <class T>
bool
do_get_sub_expr(const T &item __attribute__((unused)),
                unsigned int idx __attribute__((unused)),
                unsigned int &it __attribute__((unused)),
                const expr2tc *&ptr __attribute__((unused)))
{
  return false;
}

template <>
bool
do_get_sub_expr<expr2tc>(const expr2tc &item, unsigned int idx,
                               unsigned int &it, const expr2tc *&ptr)
{
  if (idx == it) {
    ptr = &item;
    return true;
  } else {
    it++;
    return false;
  }
}

template <>
bool
do_get_sub_expr<std::vector<expr2tc>>(const std::vector<expr2tc> &item,
                                      unsigned int idx, unsigned int &it,
                                      const expr2tc *&ptr)
{
  if (idx < it + item.size()) {
    ptr = &item[idx - it];
    return true;
  } else {
    it += item.size();
    return false;
  }
}

// Non-const versions of the above.

template <class T>
bool
do_get_sub_expr_nc(T &item __attribute__((unused)),
                unsigned int idx __attribute__((unused)),
                unsigned int &it __attribute__((unused)),
                expr2tc *&ptr __attribute__((unused)))
{
  return false;
}

template <>
bool
do_get_sub_expr_nc<expr2tc>(expr2tc &item, unsigned int idx, unsigned int &it,
                         expr2tc *&ptr)
{
  if (idx == it) {
    ptr = &item;
    return true;
  } else {
    it++;
    return false;
  }
}

template <>
bool
do_get_sub_expr_nc<std::vector<expr2tc>>(std::vector<expr2tc> &item,
                                      unsigned int idx, unsigned int &it,
                                      expr2tc *&ptr)
{
  if (idx < it + item.size()) {
    ptr = &item[idx - it];
    return true;
  } else {
    it += item.size();
    return false;
  }
}

template <class T>
unsigned int
do_count_sub_exprs(T &item __attribute__((unused)))
{
  return 0;
}

template <>
unsigned int
do_count_sub_exprs<const expr2tc>(const expr2tc &item __attribute__((unused)))
{
  return 1;
}

template <>
unsigned int
do_count_sub_exprs<const std::vector<expr2tc>>(const std::vector<expr2tc> &item)
{
  return item.size();
}

typedef std::size_t lolnoop;
inline std::size_t
hash_value(lolnoop val)
{
  return val;
}

// Local template for implementing delegate calling, with type dependency.
// Can't easily extend to cover types because field type is _already_ abstracted
template <typename T, typename U>
void
call_expr_delegate(T &ref, U &f)
{
  // Don't do anything normally.
  (void)ref;
  (void)f;
  return;
}

template <>
void
call_expr_delegate<const expr2tc,expr2t::const_op_delegate>
                  (const expr2tc &ref, expr2t::const_op_delegate &f)
{
  f(ref);
  return;
}

template <>
void
call_expr_delegate<expr2tc, expr2t::op_delegate>
                  (expr2tc &ref, expr2t::op_delegate &f)
{
  f(ref);
  return;
}

template <>
void
call_expr_delegate<const std::vector<expr2tc>, expr2t::const_op_delegate>
                 (const std::vector<expr2tc> &ref, expr2t::const_op_delegate &f)
{
  for (const expr2tc &r : ref)
    f(r);

  return;
}

template <>
void
call_expr_delegate<std::vector<expr2tc>, expr2t::op_delegate>
                  (std::vector<expr2tc> &ref, expr2t::op_delegate &f)
{
  for (expr2tc &r : ref)
    f(r);

  return;
}

// Repeat of call_expr_delegate, but for types
template <typename T, typename U>
void
call_type_delegate(T &ref, U &f)
{
  // Don't do anything normally.
  (void)ref;
  (void)f;
  return;
}

template <>
void
call_type_delegate<const type2tc,type2t::const_subtype_delegate>
                  (const type2tc &ref, type2t::const_subtype_delegate &f)
{
  f(ref);
  return;
}

template <>
void
call_type_delegate<type2tc, type2t::subtype_delegate>
                  (type2tc &ref, type2t::subtype_delegate &f)
{
  f(ref);
  return;
}

template <>
void
call_type_delegate<const std::vector<type2tc>, type2t::const_subtype_delegate>
                 (const std::vector<type2tc> &ref, type2t::const_subtype_delegate &f)
{
  for (const type2tc &r : ref)
    f(r);

  return;
}

template <>
void
call_type_delegate<std::vector<type2tc>, type2t::subtype_delegate>
                  (std::vector<type2tc> &ref, type2t::subtype_delegate &f)
{
  for (type2tc &r : ref)
    f(r);

  return;
}

/************************ Second attempt at irep templates ********************/

// Implementations of common methods, recursively.

// Top level type method definition (above recursive def)
// exprs

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
const expr2tc *
esbmct::expr_methods2<derived, baseclass, traits, container, enable, fields>::get_sub_expr(unsigned int i) const
{
  return superclass::get_sub_expr_rec(0, i); // Skips expr_id
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
expr2tc *
esbmct::expr_methods2<derived, baseclass, traits, container, enable, fields>::get_sub_expr_nc(unsigned int i)
{
  return superclass::get_sub_expr_nc_rec(0, i); // Skips expr_id
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
unsigned int
esbmct::expr_methods2<derived, baseclass, traits, container, enable, fields>::get_num_sub_exprs(void) const
{
  return superclass::get_num_sub_exprs_rec(); // Skips expr_id
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
void
esbmct::expr_methods2<derived, baseclass, traits, container, enable, fields>::foreach_operand_impl_const(expr2t::const_op_delegate &f) const
{
  superclass::foreach_operand_impl_const_rec(f);
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
void
esbmct::expr_methods2<derived, baseclass, traits, container, enable, fields>::foreach_operand_impl(expr2t::op_delegate &f)
{
  superclass::foreach_operand_impl_rec(f);
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
void
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::build_python_class(
    const typename container::id_field_type id)
{
#ifdef WITH_PYTHON
  using namespace boost::python;

  // Build python class out of the derived type (such as add2t) and with the
  // name of the expr_id. Alas, the expr id isn't currently in the type record
  // so can't be sucked out here.
  // container.

  // Certain irep names collide with python keywords. Mark those as illegal,
  // and append an underscore behind them.
  const char *basename = base_to_names<typename traits::base2t>::names[id];
  std::string basename_str(basename);
  if (std::find(illegal_python_names.begin(), illegal_python_names.end(), basename_str) != illegal_python_names.end())
    basename_str.append("_");

  class_<derived, bases<base2t>, container, boost::noncopyable>
    foo(basename_str.c_str(), no_init);

  foo.def("make", &traits::template make_contained<derived>);
  foo.staticmethod("make");

  build_python_class_rec(foo, 0);

  register_irep_methods<base2t> bar;
  bar(foo, basename);
  return;
#else
  (void) id;
#endif /* WITH_PYTHON */
}

// Types

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
void
esbmct::type_methods2<derived, baseclass, traits, container, enable, fields>::foreach_subtype_impl_const(type2t::const_subtype_delegate &f) const
{
  superclass::foreach_subtype_impl_const_rec(f);
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
void
esbmct::type_methods2<derived, baseclass, traits, container, enable, fields>::foreach_subtype_impl(type2t::subtype_delegate &f)
{
  superclass::foreach_subtype_impl_rec(f);
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
auto
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::clone(void) const -> base_container2tc
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return base_container2tc(new_obj);
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
list_of_memberst
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::tostring(unsigned int indent) const
{
  list_of_memberst thevector;

  superclass::tostring_rec(0, thevector, indent); // Skips type_id / expr_id
  return thevector;
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
bool
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::cmp(const base2t &ref) const
{
  return cmp_rec(ref); // _includes_ type_id / expr_id
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
int
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::lt(const base2t &ref) const
{
  return lt_rec(ref); // _includes_ type_id / expr_id
}

template <class derived, class baseclass, typename traits, typename container,  typename enable, typename fields>
size_t
esbmct::irep_methods2<derived, baseclass, traits, container,  enable, fields>::do_crc(size_t seed) const
{

  if (this->crc_val != 0) {
    boost::hash_combine(seed, (lolnoop)this->crc_val);
    return seed;
  }

  // Starting from 0, pass a crc value through all the sub-fields of this
  // expression. Store it into crc_val. Don't allow the input seed to affect
  // this calculation, as the crc value needs to uniquely identify _this_
  // expression.
  assert(this->crc_val == 0);

  do_crc_rec(); // _includes_ type_id / expr_id

  // Finally, combine the crc of this expr with the input seed, and return
  boost::hash_combine(seed, (lolnoop)this->crc_val);
  return seed;
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
void
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::hash(crypto_hash &hash) const
{

  hash_rec(hash); // _includes_ type_id / expr_id
  return;
}

// The, *actual* recursive defs

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
void
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::tostring_rec(unsigned int idx, list_of_memberst &vec, unsigned int indent) const
{
  // Skip over type fields in expressions. Alas, this is a design oversight,
  // without this we would screw up the field name list.
  // It escapes me why this isn't printed here anyway, it gets printed in the
  // end.
  if (std::is_same<cur_type, type2tc>::value && std::is_base_of<expr2t,derived>::value) {
    superclass::tostring_rec(idx, vec, indent);
    return;
  }

  // Insert our particular member to string list.
  const derived *derived_this = static_cast<const derived*>(this);
  auto m_ptr = membr_ptr::value;
  do_type2string<cur_type>(derived_this->*m_ptr, idx, derived_this->field_names, vec, indent);

  // Recurse
  superclass::tostring_rec(idx + 1, vec, indent);
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
bool
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::cmp_rec(const base2t &ref) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  const derived *ref2 = static_cast<const derived *>(&ref);
  auto m_ptr = membr_ptr::value;

  if (!do_type_cmp(derived_this->*m_ptr, ref2->*m_ptr))
    return false;

  return superclass::cmp_rec(ref);
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
int
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::lt_rec(const base2t &ref) const
{
  int tmp;
  const derived *derived_this = static_cast<const derived*>(this);
  const derived *ref2 = static_cast<const derived *>(&ref);
  auto m_ptr = membr_ptr::value;

  tmp = do_type_lt(derived_this->*m_ptr, ref2->*m_ptr);
  if (tmp != 0)
    return tmp;

  return superclass::lt_rec(ref);
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
void
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::do_crc_rec() const
{
  const derived *derived_this = static_cast<const derived*>(this);
  auto m_ptr = membr_ptr::value;

  size_t tmp = do_type_crc(derived_this->*m_ptr, this->crc_val);
  boost::hash_combine(this->crc_val, (lolnoop)tmp);

  superclass::do_crc_rec();
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
void
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::hash_rec(crypto_hash &hash) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  auto m_ptr = membr_ptr::value;
  do_type_hash(derived_this->*m_ptr, hash);

  superclass::hash_rec(hash);
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
const expr2tc *
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::get_sub_expr_rec(unsigned int cur_idx, unsigned int desired) const
{
  const expr2tc *ptr;
  const derived *derived_this = static_cast<const derived*>(this);
  auto m_ptr = membr_ptr::value;

  // XXX -- this takes a _reference_ to cur_idx, and maybe modifies.
  if (do_get_sub_expr(derived_this->*m_ptr, desired, cur_idx, ptr))
    return ptr;

  return superclass::get_sub_expr_rec(cur_idx, desired);
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
expr2tc *
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::get_sub_expr_nc_rec(unsigned int cur_idx, unsigned int desired)
{
  expr2tc *ptr;
  derived *derived_this = static_cast<derived*>(this);
  auto m_ptr = membr_ptr::value;

  // XXX -- this takes a _reference_ to cur_idx, and maybe modifies.
  if (do_get_sub_expr_nc(derived_this->*m_ptr, desired, cur_idx, ptr))
    return ptr;

  return superclass::get_sub_expr_nc_rec(cur_idx, desired);
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
unsigned int
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::get_num_sub_exprs_rec(void) const
{
  unsigned int num = 0;
  const derived *derived_this = static_cast<const derived*>(this);
  auto m_ptr = membr_ptr::value;

  num = do_count_sub_exprs(derived_this->*m_ptr);
  return num + superclass::get_num_sub_exprs_rec();
}

// Operand iteration specialized for expr2tc: call delegate.
template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
void
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::foreach_operand_impl_const_rec(expr2t::const_op_delegate &f) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  auto m_ptr = membr_ptr::value;

  // Call delegate
  call_expr_delegate(derived_this->*m_ptr, f);

  superclass::foreach_operand_impl_const_rec(f);
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
void
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::foreach_operand_impl_rec(expr2t::op_delegate &f)
{
  derived *derived_this = static_cast<derived*>(this);
  auto m_ptr = membr_ptr::value;

  // Call delegate
  call_expr_delegate(derived_this->*m_ptr, f);

  superclass::foreach_operand_impl_rec(f);
}

#ifdef WITH_PYTHON

// Misery: the field we work with at a particular point in the inheretance
// chain may be an expr_id, type_id, or expr2t::type, all of which have their
// own special cases elsewhere. More importantly, the field_names array only
// starts _after_ those fields. So we can't generically add them to the python
// class via this method.
// Rather than re-juggling all of these things, just specialize for those right
// now. This is not tidy, but never mind.
// Better: we can't pass field_traits::value into this, because the compiler
// requires that the direct syntax &X::Y is passed in, not a value equivalent
// to it. Bah.

template <typename R, typename T>
class field_to_be_skipped
{
  public:
  static bool value(R T::*foo __attribute__((unused))) { return false; }
};

// Specialise for those fields we skip.
template<> class field_to_be_skipped<type2t::type_ids, type2t>
{public: static bool value(type2t::type_ids type2t::*foo) { if (foo == &type2t::type_id) return true; else return false; } };
template<> class field_to_be_skipped<const expr2t::expr_ids, expr2t>
{public: static bool value(const expr2t::expr_ids expr2t::*foo) { if (foo == &expr2t::expr_id) return true; else return false; } };
template<> class field_to_be_skipped<type2tc, expr2t>
{public: static bool value(type2tc expr2t::*foo) { if (foo == &expr2t::type) return true; else return false; } };

#endif /* WITH PYTHON */

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
template <typename T>
void
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::build_python_class_rec(T &obj, unsigned int idx)
{
#ifdef WITH_PYTHON
  // Optionally skip this field if it's generic to types / exprs.
  if (field_to_be_skipped<cur_type, base_class>::value(membr_ptr::value)) {
    assert(idx == 0);
    superclass::build_python_class_rec(obj, idx);
    return;
  }

  // Add this field record to the python class obj, get name from field_names
  // field, and increment the index we're working on.
  superclass::build_python_class_rec(
      obj.def_readonly(derived::field_names[idx].c_str(), membr_ptr::value), idx+1);
  return;
#else
  (void) obj;
  (void) idx;
#endif
}


template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
void
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::foreach_subtype_impl_const_rec(type2t::const_subtype_delegate &f) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  auto m_ptr = membr_ptr::value;

  // Call delegate
  call_type_delegate(derived_this->*m_ptr, f);

  superclass::foreach_subtype_impl_const_rec(f);
}

template <class derived, class baseclass, typename traits, typename container, typename enable, typename fields>
void
esbmct::irep_methods2<derived, baseclass, traits, container, enable, fields>::foreach_subtype_impl_rec(type2t::subtype_delegate &f)
{
  derived *derived_this = static_cast<derived*>(this);
  auto m_ptr = membr_ptr::value;

  // Call delegate
  call_type_delegate(derived_this->*m_ptr, f);

  superclass::foreach_subtype_impl_rec(f);
}

/********************** Constants and explicit instantiations *****************/

std::string bool_type2t::field_names [esbmct::num_type_fields]  = {"","","","", ""};
std::string empty_type2t::field_names [esbmct::num_type_fields]  =
{ "", "", "", "", ""};
std::string symbol_type2t::field_names [esbmct::num_type_fields]  =
{ "symbol_name", "", "", "", ""};
std::string struct_type2t::field_names [esbmct::num_type_fields]  =
{ "members", "member_names", "member_pretty_names", "typename", "", ""};
std::string union_type2t::field_names [esbmct::num_type_fields]  =
{ "members", "member_names", "member_pretty_names", "typename", "", ""};
std::string unsignedbv_type2t::field_names [esbmct::num_type_fields]  =
{ "width", "", "", "", ""};
std::string signedbv_type2t::field_names [esbmct::num_type_fields]  =
{ "width", "", "", "", ""};
std::string code_type2t::field_names [esbmct::num_type_fields]  =
{ "arguments", "ret_type", "argument_names", "ellipsis", ""};
std::string array_type2t::field_names [esbmct::num_type_fields]  =
{ "subtype", "array_size", "size_is_infinite", "", ""};
std::string pointer_type2t::field_names [esbmct::num_type_fields]  =
{ "subtype", "", "", "", ""};
std::string fixedbv_type2t::field_names [esbmct::num_type_fields]  =
{ "width", "integer_bits", "", "", ""};
std::string floatbv_type2t::field_names [esbmct::num_type_fields]  =
{ "fraction", "exponent", "", "", ""};
std::string string_type2t::field_names [esbmct::num_type_fields]  =
{ "width", "", "", "", ""};
std::string cpp_name_type2t::field_names [esbmct::num_type_fields]  =
{ "name", "template args", "", "", ""};

// Exprs

std::string constant_int2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string constant_fixedbv2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string constant_floatbv2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string constant_struct2t::field_names [esbmct::num_type_fields]  =
{ "members", "", "", "", ""};
std::string constant_union2t::field_names [esbmct::num_type_fields]  =
{ "members", "", "", "", ""};
std::string constant_bool2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string constant_array2t::field_names [esbmct::num_type_fields]  =
{ "members", "", "", "", ""};
std::string constant_array_of2t::field_names [esbmct::num_type_fields]  =
{ "initializer", "", "", "", ""};
std::string constant_string2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string symbol2t::field_names [esbmct::num_type_fields]  =
{ "name", "renamelev", "level1_num", "level2_num", "thread_num", "node_num"};
std::string typecast2t::field_names [esbmct::num_type_fields]  =
{ "from", "rounding_mode", "", "", "", ""};
std::string bitcast2t::field_names [esbmct::num_type_fields]  =
{ "from", "rounding_mode", "", "", "", ""};
std::string nearbyint2t::field_names [esbmct::num_type_fields]  =
{ "from", "rounding_mode", "", "", "", ""};
std::string if2t::field_names [esbmct::num_type_fields]  =
{ "cond", "true_value", "false_value", "", ""};
std::string equality2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string notequal2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string lessthan2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string greaterthan2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string lessthanequal2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string greaterthanequal2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string not2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string and2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string or2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string xor2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string implies2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string bitand2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string bitor2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string bitxor2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string bitnand2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string bitnor2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string bitnxor2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string lshr2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string bitnot2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string neg2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string abs2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string add2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string sub2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string mul2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string div2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string ieee_add2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "rounding_mode", "", "", ""};
std::string ieee_sub2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "rounding_mode", "", "", ""};
std::string ieee_mul2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "rounding_mode", "", "", ""};
std::string ieee_div2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "rounding_mode", "", "", ""};
std::string ieee_fma2t::field_names [esbmct::num_type_fields]  =
{ "value_1", "value_2", "value_3", "rounding_mode", "", ""};
std::string modulus2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string shl2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string ashr2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string same_object2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string pointer_offset2t::field_names [esbmct::num_type_fields]  =
{ "pointer_obj", "", "", "", ""};
std::string pointer_object2t::field_names [esbmct::num_type_fields]  =
{ "pointer_obj", "", "", "", ""};
std::string address_of2t::field_names [esbmct::num_type_fields]  =
{ "pointer_obj", "", "", "", ""};
std::string byte_extract2t::field_names [esbmct::num_type_fields]  =
{ "source_value", "source_offset", "big_endian", "", ""};
std::string byte_update2t::field_names [esbmct::num_type_fields]  =
{ "source_value", "source_offset", "update_value", "big_endian", ""};
std::string with2t::field_names [esbmct::num_type_fields]  =
{ "source_value", "update_field", "update_value", "", ""};
std::string member2t::field_names [esbmct::num_type_fields]  =
{ "source_value", "member_name", "", "", ""};
std::string index2t::field_names [esbmct::num_type_fields]  =
{ "source_value", "index", "", "", ""};
std::string isnan2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string overflow2t::field_names [esbmct::num_type_fields]  =
{ "operand", "", "", "", ""};
std::string overflow_cast2t::field_names [esbmct::num_type_fields]  =
{ "operand", "bits", "", "", ""};
std::string overflow_neg2t::field_names [esbmct::num_type_fields]  =
{ "operand", "", "", "", ""};
std::string unknown2t::field_names [esbmct::num_type_fields]  =
{ "", "", "", "", ""};
std::string invalid2t::field_names [esbmct::num_type_fields]  =
{ "", "", "", "", ""};
std::string null_object2t::field_names [esbmct::num_type_fields]  =
{ "", "", "", "", ""};
std::string dynamic_object2t::field_names [esbmct::num_type_fields]  =
{ "instance", "invalid", "unknown", "", ""};
std::string dereference2t::field_names [esbmct::num_type_fields]  =
{ "pointer", "", "", "", ""};
std::string valid_object2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string deallocated_obj2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string dynamic_size2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string sideeffect2t::field_names [esbmct::num_type_fields]  =
{ "operand", "size", "arguments", "alloctype", "kind"};
std::string code_block2t::field_names [esbmct::num_type_fields]  =
{ "operands", "", "", "", ""};
std::string code_assign2t::field_names [esbmct::num_type_fields]  =
{ "target", "source", "", "", ""};
std::string code_init2t::field_names [esbmct::num_type_fields]  =
{ "target", "source", "", "", ""};
std::string code_decl2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string code_printf2t::field_names [esbmct::num_type_fields]  =
{ "operands", "", "", "", ""};
std::string code_expression2t::field_names [esbmct::num_type_fields]  =
{ "operand", "", "", "", ""};
std::string code_return2t::field_names [esbmct::num_type_fields]  =
{ "operand", "", "", "", ""};
std::string code_skip2t::field_names [esbmct::num_type_fields]  =
{ "", "", "", "", ""};
std::string code_free2t::field_names [esbmct::num_type_fields]  =
{ "operand", "", "", "", ""};
std::string code_goto2t::field_names [esbmct::num_type_fields]  =
{ "target", "", "", "", ""};
std::string object_descriptor2t::field_names [esbmct::num_type_fields]  =
{ "object", "offset", "alignment", "", ""};
std::string code_function_call2t::field_names [esbmct::num_type_fields]  =
{ "return_sym", "function", "operands", "", ""};
std::string code_comma2t::field_names [esbmct::num_type_fields]  =
{ "side_1", "side_2", "", "", ""};
std::string invalid_pointer2t::field_names [esbmct::num_type_fields]  =
{ "pointer_obj", "", "", "", ""};
std::string code_asm2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string code_cpp_del_array2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string code_cpp_delete2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string code_cpp_catch2t::field_names [esbmct::num_type_fields]  =
{ "exception_list", "", "", "", ""};
std::string code_cpp_throw2t::field_names [esbmct::num_type_fields]  =
{ "operand", "exception_list", "", "", ""};
std::string code_cpp_throw_decl2t::field_names [esbmct::num_type_fields]  =
{ "exception_list", "", "", "", ""};
std::string code_cpp_throw_decl_end2t::field_names [esbmct::num_type_fields]  =
{ "exception_list", "", "", "", ""};
std::string isinf2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string isnormal2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string isfinite2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string signbit2t::field_names [esbmct::num_type_fields]  =
{ "value", "", "", "", ""};
std::string concat2t::field_names [esbmct::num_type_fields]  =
{ "forward", "aft", "", "", ""};

// This has become particularly un-fun with the arrival of gcc 6.x and clang
// 3.8 (roughly). Both are very aggressive wrt. whether templates are actually
// instantiated or not, and refuse to instantiate template base classes
// implicitly. Unfortunately, we relied on that before; it might still be
// instantiating some of the irep_methods2 chain, but it's not doing the
// method definitions, which leads to linking failures later.
//
// I've experimented with a variety of ways to implicitly require each method
// of our template chain, but none seem to succeed, and the compiler goes a
// long way out of it's path to avoid these instantiations. The real real
// issue seems to be virtual functions, the compiler can jump through many
// hoops to get method addresses out of the vtable, rather than having to
// implicitly define it. One potential workaround may be a non-virtual method
// that gets defined that calls all virtual methods explicitly?
//
// Anyway: the workaround is to explicitly instantiate each level of the
// irep_methods2 hierarchy, with associated pain and suffering. This means
// that our template system isn't completely variadic, you have to know how
// many levels to instantiate when you reach this level, explicitly. Which
// sucks, but is a small price to pay.

#undef irep_typedefs
#undef irep_typedefs_empty

#define irep_typedefs0(basename, superclass) \
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename superclass::traits::fields>::type>;

#define irep_typedefs1(basename, superclass) \
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename superclass::traits::fields>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>;

#define irep_typedefs2(basename, superclass) \
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename superclass::traits::fields>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>;

#define irep_typedefs3(basename, superclass) \
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename superclass::traits::fields>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>::type>;

#define irep_typedefs4(basename, superclass) \
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename superclass::traits::fields>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>::type>::type>;

#define irep_typedefs5(basename, superclass) \
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename superclass::traits::fields>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>::type>::type>::type>;

#define irep_typedefs6(basename, superclass) \
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename superclass::traits::fields>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>::type>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>::type>::type>::type>::type>;

////////////////////////////

#define type_typedefs1(basename, superclass) \
  template class esbmct::type_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc, boost::mpl::pop_front<typename superclass::traits::fields>::type>;

#define type_typedefs2(basename, superclass) \
  template class esbmct::type_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc, boost::mpl::pop_front<typename superclass::traits::fields>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>;

#define type_typedefs3(basename, superclass) \
  template class esbmct::type_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc, boost::mpl::pop_front<typename superclass::traits::fields>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>;

#define type_typedefs4(basename, superclass) \
  template class esbmct::type_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc, boost::mpl::pop_front<typename superclass::traits::fields>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>;\
  template class esbmct::irep_methods2<basename##2t, superclass, typename superclass::traits, basename##2tc, boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename boost::mpl::pop_front<typename superclass::traits::fields>::type>::type>::type>::type>;

#define type_typedefs_empty(basename)\
  template class esbmct::type_methods2<basename##2t, type2t, esbmct::type2t_default_traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, type2t, esbmct::type2t_default_traits, basename##2tc>;

type_typedefs_empty(bool_type)
type_typedefs_empty(empty_type)
type_typedefs1(symbol_type, symbol_type_data)
type_typedefs4(struct_type, struct_union_data)
type_typedefs4(union_type, struct_union_data)
type_typedefs1(unsignedbv_type, bv_data)
type_typedefs1(signedbv_type, bv_data)
type_typedefs4(code_type, code_data)
type_typedefs3(array_type, array_data)
type_typedefs1(pointer_type, pointer_data)
type_typedefs2(fixedbv_type, fixedbv_data)
type_typedefs2(floatbv_type, floatbv_data)
type_typedefs1(string_type, string_data)
type_typedefs2(cpp_name_type, cpp_name_data)

// Explicit instanciation for exprs.

#define expr_typedefs1(basename, superclass) \
  template class esbmct::expr_methods2<basename##2t, superclass, superclass::traits, basename##2tc>;\
  irep_typedefs1(basename, superclass)

#define expr_typedefs2(basename, superclass) \
  template class esbmct::expr_methods2<basename##2t, superclass, superclass::traits, basename##2tc>;\
  irep_typedefs2(basename, superclass)

#define expr_typedefs3(basename, superclass) \
  template class esbmct::expr_methods2<basename##2t, superclass, superclass::traits, basename##2tc>;\
  irep_typedefs3(basename, superclass)

#define expr_typedefs4(basename, superclass) \
  template class esbmct::expr_methods2<basename##2t, superclass, superclass::traits, basename##2tc>;\
  irep_typedefs4(basename, superclass)

#define expr_typedefs5(basename, superclass) \
  template class esbmct::expr_methods2<basename##2t, superclass, superclass::traits, basename##2tc>;\
  irep_typedefs5(basename, superclass)

#define expr_typedefs6(basename, superclass) \
  template class esbmct::expr_methods2<basename##2t, superclass, superclass::traits, basename##2tc>;\
  irep_typedefs6(basename, superclass)

#define expr_typedefs_empty(basename, superclass) \
  template class esbmct::expr_methods2<basename##2t, superclass, esbmct::expr2t_default_traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, superclass, superclass::traits, basename##2tc>;\
  template class esbmct::irep_methods2<basename##2t, superclass, esbmct::expr2t_default_traits, basename##2tc, boost::mpl::pop_front<typename superclass::traits::fields>::type>;

expr_typedefs1(constant_int, constant_int_data);
expr_typedefs1(constant_fixedbv, constant_fixedbv_data);
expr_typedefs1(constant_floatbv, constant_floatbv_data);
expr_typedefs1(constant_struct, constant_datatype_data);
expr_typedefs1(constant_union, constant_datatype_data);
expr_typedefs1(constant_array, constant_datatype_data);
expr_typedefs1(constant_bool, constant_bool_data);
expr_typedefs1(constant_array_of, constant_array_of_data);
expr_typedefs1(constant_string, constant_string_data);
expr_typedefs6(symbol, symbol_data);
expr_typedefs2(nearbyint, typecast_data);
expr_typedefs2(typecast,typecast_data);
expr_typedefs2(bitcast,typecast_data);
expr_typedefs3(if, if_data);
expr_typedefs2(equality, relation_data);
expr_typedefs2(notequal, relation_data);
expr_typedefs2(lessthan, relation_data);
expr_typedefs2(greaterthan, relation_data);
expr_typedefs2(lessthanequal, relation_data);
expr_typedefs2(greaterthanequal, relation_data);
expr_typedefs1(not, bool_1op);
expr_typedefs2(and, logic_2ops);
expr_typedefs2(or, logic_2ops);
expr_typedefs2(xor, logic_2ops);
expr_typedefs2(implies, logic_2ops);
expr_typedefs2(bitand, bit_2ops);
expr_typedefs2(bitor, bit_2ops);
expr_typedefs2(bitxor, bit_2ops);
expr_typedefs2(bitnand, bit_2ops);
expr_typedefs2(bitnor, bit_2ops);
expr_typedefs2(bitnxor, bit_2ops);
expr_typedefs2(lshr, bit_2ops);
expr_typedefs1(bitnot, bitnot_data);
expr_typedefs1(neg, arith_1op);
expr_typedefs1(abs, arith_1op);
expr_typedefs2(add, arith_2ops);
expr_typedefs2(sub, arith_2ops);
expr_typedefs2(mul, arith_2ops);
expr_typedefs2(div, arith_2ops);
expr_typedefs3(ieee_add, ieee_arith_2ops);
expr_typedefs3(ieee_sub, ieee_arith_2ops);
expr_typedefs3(ieee_mul, ieee_arith_2ops);
expr_typedefs3(ieee_div, ieee_arith_2ops);
expr_typedefs4(ieee_fma, ieee_arith_3ops);
expr_typedefs2(modulus, arith_2ops);
expr_typedefs2(shl, arith_2ops);
expr_typedefs2(ashr, arith_2ops);
expr_typedefs2(same_object, same_object_data);
expr_typedefs1(pointer_offset, pointer_ops);
expr_typedefs1(pointer_object, pointer_ops);
expr_typedefs1(address_of, pointer_ops);
expr_typedefs3(byte_extract, byte_extract_data);
expr_typedefs4(byte_update, byte_update_data);
expr_typedefs3(with, with_data);
expr_typedefs2(member, member_data);
expr_typedefs2(index, index_data);
expr_typedefs1(isnan, bool_1op);
expr_typedefs1(overflow, overflow_ops);
expr_typedefs2(overflow_cast, overflow_cast_data);
expr_typedefs1(overflow_neg, overflow_ops);
expr_typedefs_empty(unknown, expr2t);
expr_typedefs_empty(invalid, expr2t);
expr_typedefs_empty(null_object, expr2t);
expr_typedefs3(dynamic_object, dynamic_object_data);
expr_typedefs2(dereference, dereference_data);
expr_typedefs1(valid_object, object_ops);
expr_typedefs1(deallocated_obj, object_ops);
expr_typedefs1(dynamic_size, object_ops);
expr_typedefs5(sideeffect, sideeffect_data);
expr_typedefs1(code_block, code_block_data);
expr_typedefs2(code_assign, code_assign_data);
expr_typedefs2(code_init, code_assign_data);
expr_typedefs1(code_decl, code_decl_data);
expr_typedefs1(code_printf, code_printf_data);
expr_typedefs1(code_expression, code_expression_data);
expr_typedefs1(code_return, code_expression_data);
expr_typedefs_empty(code_skip, expr2t);
expr_typedefs1(code_free, code_expression_data);
expr_typedefs1(code_goto, code_goto_data);
expr_typedefs3(object_descriptor, object_desc_data);
expr_typedefs3(code_function_call, code_funccall_data);
expr_typedefs2(code_comma, code_comma_data);
expr_typedefs1(invalid_pointer, invalid_pointer_ops);
expr_typedefs1(code_asm, code_asm_data);
expr_typedefs1(code_cpp_del_array, code_expression_data);
expr_typedefs1(code_cpp_delete, code_expression_data);
expr_typedefs1(code_cpp_catch, code_cpp_catch_data);
expr_typedefs2(code_cpp_throw, code_cpp_throw_data);
expr_typedefs2(code_cpp_throw_decl, code_cpp_throw_decl_data);
expr_typedefs1(code_cpp_throw_decl_end, code_cpp_throw_decl_data);
expr_typedefs1(isinf, bool_1op);
expr_typedefs1(isnormal, bool_1op);
expr_typedefs1(isfinite, bool_1op);
expr_typedefs1(signbit, overflow_ops);
expr_typedefs2(concat, bit_2ops);
