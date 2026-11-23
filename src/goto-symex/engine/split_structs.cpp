#include <goto-symex/engine/split_structs.h>
#include <irep2/irep2_utils.h>
#include <pointer-analysis/andersen.h>
#include <util/base/prefix.h>
#include <util/config/config.h>
#include <util/message/message.h>

namespace
{
/* Collects the struct objects a program names and drops each one that some
 * use could not take as the struct of its fields. */
class split_scant
{
public:
  std::unordered_map<irep_idt, type2tc, irep_id_hash> seen;
  std::unordered_set<irep_idt, irep_id_hash> excluded;
  /* Objects whose address is taken as `&s`; pointer_scant decides whether
   * every access through that address keeps a field path. */
  std::unordered_set<irep_idt, irep_id_hash> addressed;

  /* How a use under the current node may reach an object: any use escapes
   * under an address or a side effect, and a whole one under a byte-level or
   * reinterpreting operation. */
  enum class uset
  {
    plain,
    whole_escapes,
    any_escapes
  };

  /* The use the operands of @p e inherit from a use @p use of @p e. */
  static uset use_under(const expr2tc &e, uset use)
  {
    if (
      is_address_of2t(e) ||
      (is_sideeffect2t(e) &&
       to_sideeffect2t(e).kind != sideeffect2t::allockind::nondet))
      return uset::any_escapes;
    const bool reinterprets = is_byte_extract2t(e) || is_byte_update2t(e) ||
                              is_bitcast2t(e) || is_typecast2t(e);
    return use == uset::plain && reinterprets ? uset::whole_escapes : use;
  }

  /* Records the object @p sym names; one read at two types stays whole. */
  irep_idt see(const expr2tc &sym)
  {
    const auto [it, fresh] = seen.emplace(to_symbol2t(sym).thename, sym->type);
    if (!fresh && it->second != sym->type)
      excluded.insert(it->first);
    return it->first;
  }

  /* One walk over @p e; a member of a symbol is not a use of the whole. */
  void note(const expr2tc &e, uset use = uset::plain)
  {
    if (is_nil_expr(e))
      return;
    if (
      use == uset::plain && is_address_of2t(e) &&
      is_symbol2t(to_address_of2t(e).ptr_obj))
    {
      addressed.insert(see(to_address_of2t(e).ptr_obj));
      return;
    }
    use = use_under(e, use);

    if (is_symbol2t(e))
    {
      const irep_idt name = see(e);
      if (use != uset::plain)
        excluded.insert(name);
      return;
    }
    if (is_member2t(e) && is_symbol2t(to_member2t(e).source_value))
    {
      note(
        to_member2t(e).source_value,
        use == uset::any_escapes ? use : uset::plain);
      return;
    }
    e->foreach_operand([this, use](const expr2tc &op) { note(op, use); });
  }

  void analyse(const goto_programt::instructiont &i)
  {
    note(i.guard);
    for (const expr2tc &inv : i.get_loop_invariants())
      note(inv, uset::any_escapes);
    for (const expr2tc &target : i.get_loop_assigns_targets())
      note(target, uset::any_escapes);
    if (is_nil_expr(i.code))
      return;

    /* The struct of an object's fields can be written only as the whole
     * target of an assignment or call; anywhere else in a target it is not
     * an lvalue. */
    const expr2tc *target = nullptr;
    if (is_code_function_call2t(i.code))
      target = &to_code_function_call2t(i.code).ret;
    else if (is_code_assign2t(i.code))
      target = &to_code_assign2t(i.code).target;
    if (!target || is_nil_expr(*target) || is_symbol2t(*target))
    {
      note(i.code);
      return;
    }
    i.code->foreach_operand([&](const expr2tc &op) {
      note(op, op.get() == target->get() ? uset::whole_escapes : uset::plain);
    });
  }
};

/* Drops each addressed candidate some access through a pointer could reach
 * other than as the whole object or one of its members, both of the object's
 * own type, since symex would then build that access at byte level. Andersen's
 * points-to sets are sound, so a pointer it cannot bound drops every one. */
class pointer_scant
{
public:
  pointer_scant(
    andersent &points_to,
    std::unordered_map<irep_idt, type2tc, irep_id_hash> candidates)
    : candidates(std::move(candidates)), points_to(points_to)
  {
  }

  std::unordered_map<irep_idt, type2tc, irep_id_hash> candidates;

  void
  analyse(const goto_functionst &goto_functions, goto_programt::const_targett i)
  {
    loc = i;
    walk(i->guard, nullptr, nullptr);
    if (is_nil_expr(i->code))
      return;
    if (
      i->is_other() ||
      (i->is_function_call() && opaque_call(goto_functions, i->code)))
      i->code->foreach_operand([this](const expr2tc &op) { drop_reached(op); });
    walk(i->code, nullptr, nullptr);
  }

private:
  andersent &points_to;
  goto_programt::const_targett loc;

  /* Symex runs these itself, or havocs what they reach, without a dereference
   * this scan can see. */
  static bool
  opaque_call(const goto_functionst &goto_functions, const expr2tc &code)
  {
    const expr2tc &f = to_code_function_call2t(code).function;
    if (!is_symbol2t(f))
      return true;
    const std::string &id = to_symbol2t(f).thename.as_string();
    if (
      has_prefix(id, "c:@F@__ESBMC") || has_prefix(id, "c:@F@__builtin") ||
      id == "c:@F@scanf" || id == "c:@F@sscanf" || id == "c:@F@fscanf")
      return true;
    auto it = goto_functions.function_map.find(to_symbol2t(f).thename);
    return it == goto_functions.function_map.end() ||
           !it->second.body_available;
  }

  /* The candidates @p pointer may point to. One Andersen cannot bound may
   * point into any of them at any offset, so it drops them all. */
  std::vector<irep_idt> reached(const expr2tc &pointer)
  {
    value_setst::valuest objects;
    points_to.get_values(loc, pointer, objects);
    std::vector<irep_idt> hit;
    for (const expr2tc &v : objects)
    {
      if (!is_object_descriptor2t(v))
      {
        candidates.clear();
        return {};
      }
      const expr2tc &obj = to_object_descriptor2t(v).object;
      if (is_symbol2t(obj) && candidates.count(to_symbol2t(obj).thename))
        hit.push_back(to_symbol2t(obj).thename);
    }
    return hit;
  }

  void drop_reached(const expr2tc &e)
  {
    if (is_nil_expr(e) || candidates.empty() || !is_pointer_type(e))
      return;
    for (const irep_idt &n : reached(e))
      candidates.erase(n);
  }

  static bool keeps_field_path(const expr2tc *parent, const expr2tc *grand)
  {
    auto reinterprets_or_offsets = [](const expr2tc *e) {
      return e && (is_index2t(*e) || is_address_of2t(*e) || is_member2t(*e) ||
                   is_byte_extract2t(*e) || is_byte_update2t(*e) ||
                   is_bitcast2t(*e) || is_with2t(*e));
    };
    if (parent && is_member2t(*parent))
      return !reinterprets_or_offsets(grand);
    return !reinterprets_or_offsets(parent);
  }

  void walk(const expr2tc &e, const expr2tc *parent, const expr2tc *grand)
  {
    if (is_nil_expr(e) || candidates.empty())
      return;
    if (is_dereference2t(e))
    {
      const bool ok = keeps_field_path(parent, grand);
      // Andersen may list one object more than once.
      for (const irep_idt &n : reached(to_dereference2t(e).value))
      {
        auto c = candidates.find(n);
        if (c != candidates.end() && (!ok || c->second != e->type))
          candidates.erase(c);
      }
    }
    else if ((is_add2t(e) || is_sub2t(e)) && is_pointer_type(e->type))
      drop_reached(e);
    e->foreach_operand([&](const expr2tc &op) { walk(op, &e, parent); });
  }
};

/* The objects @p scan saw addressed that every access through a pointer
 * reaches as a whole or through one member. */
std::unordered_map<irep_idt, type2tc, irep_id_hash> pointer_accessible(
  const goto_functionst &goto_functions,
  const split_scant &scan)
{
  std::unordered_map<irep_idt, type2tc, irep_id_hash> candidates;
  for (const irep_idt &name : scan.addressed)
    if (!scan.excluded.count(name))
      candidates.emplace(name, scan.seen.at(name));
  if (candidates.empty())
    return candidates;

  andersent points_to;
  points_to(goto_functions);
  pointer_scant pointers(points_to, std::move(candidates));
  // Andersen gives an uncalled function's pointer parameters TOP.
  for (const irep_idt &fn : reachable_functions(goto_functions))
  {
    auto f = goto_functions.function_map.find(fn);
    if (f == goto_functions.function_map.end() || !f->second.body_available)
      continue;
    const goto_programt &body = f->second.body;
    for (auto i = body.instructions.begin(); i != body.instructions.end(); ++i)
      pointers.analyse(goto_functions, i);
  }
  return std::move(pointers.candidates);
}

/* A flexible, zero-length or variable-length array member stays in its
 * struct. */
bool layable(const type2tc &t)
{
  if (!is_array_type(t))
    return true;
  const array_type2t &a = to_array_type(t);
  return !a.size_is_infinite && is_constant_int2t(a.array_size) &&
         !to_constant_int2t(a.array_size).value.is_zero();
}

/* A static object stays whole when it is per-thread, or when the
 * memory-leak check reads it as a root by its own name. */
bool splittable(const symbolt &s, const type2tc &type, bool leak_roots)
{
  if (
    s.is_extern || s.mode == "Python" || !is_struct_type(type) ||
    (s.static_lifetime && (leak_roots || s.is_thread_local)))
    return false;
  const struct_type2t &st = to_struct_type(type);
  return !st.members.empty() &&
         std::all_of(st.members.begin(), st.members.end(), layable);
}
} // namespace

bool split_structst::enabled(
  const goto_functionst &goto_functions,
  const optionst &options)
{
  if (
    !options.get_bool_option("field-sensitive-ssa") ||
    atol(options.get_option("stack-limit").c_str()) > 0 ||
    atol(options.get_option("total-stack-limit").c_str()) > 0)
    return false;
  if (spawns_threads(goto_functions))
  {
    log_warning(
      "--field-sensitive-ssa is ignored: the program may create threads");
    return false;
  }
  // These read program inputs off the SSA steps that read a nondet symbol,
  // which a split nondet write spreads over its fields.
  for (const char *reads_inputs :
       {"all-witnesses",
        "generate-testcase",
        "generate-pytest-testcase",
        "generate-ctest-testcase"})
    if (options.get_bool_option(reads_inputs))
    {
      log_warning("--field-sensitive-ssa is ignored with --{}", reads_inputs);
      return false;
    }
  return true;
}

void split_structst::add_fields(
  const symbolt &s,
  const struct_type2t &st,
  contextt &context)
{
  std::vector<expr2tc> &fields = split[s.id];
  auto &member_index = index[s.id];
  for (size_t k = 0; k < st.members.size(); k++)
  {
    member_index.emplace(st.member_names[k], k);
    const std::string id = id2string(s.id) + "::" + std::to_string(k);
    fields.push_back(symbol2tc(st.members[k], id));
    field_ids.insert(id);
    if (is_untracked_scalar(st.members[k]))
      untracked_fields.insert(id);
    if (context.find_symbol(id))
      continue;
    symbolt var = s;
    var.id = id;
    var.name = id2string(s.name) + "." + id2string(st.member_pretty_names[k]);
    var.is_parameter = false;
    var.set_type(st.members[k]);
    var.set_value(expr2tc());
    context.add(var);
  }
}

void split_structst::analyse(
  const goto_functionst &goto_functions,
  const namespacet &ns,
  contextt &context,
  const optionst &options,
  const std::unordered_set<irep_idt, irep_id_hash> &address_taken_locals)
{
  split.clear();
  index.clear();
  untracked_fields.clear();
  field_ids.clear();
  opaque.clear();
  if (!enabled(goto_functions, options))
    return;

  split_scant scan;
  for (const auto &[name, f] : goto_functions.function_map)
    if (f.body_available)
      for (const auto &i : f.body.instructions)
        scan.analyse(i);

  const bool leak_roots = options.get_bool_option("memory-leak-check");
  const auto by_field_paths = pointer_accessible(goto_functions, scan);
  for (const auto &[name, type] : scan.seen)
  {
    if (
      scan.excluded.count(name) ||
      ((scan.addressed.count(name) || address_taken_locals.count(name)) &&
       !by_field_paths.count(name)))
      continue;
    const symbolt *s = ns.lookup(name);
    if (!s || !splittable(*s, type, leak_roots))
      continue;

    add_fields(*s, to_struct_type(type), context);
  }
  if (!split.empty())
    log_status("Field-sensitive SSA splits {} struct object(s)", split.size());
}

bool split_structst::is_untracked_scalar(const type2tc &type)
{
  if (is_bool_type(type) || is_floatbv_type(type) || is_fixedbv_type(type))
    return true;
  // value_sett records provenance on an integer wide enough for an address.
  return (is_signedbv_type(type) || is_unsignedbv_type(type)) &&
         type->get_width() < config.ansi_c.pointer_width();
}

/* @p field at the renaming level of @p sym. */
static expr2tc at_level_of(const expr2tc &field, const symbol2t &sym)
{
  if (sym.rlevel == symbol_renaming_level::level0)
    return field;
  // Fields only ever take their object's L1 activation, never an L2 name.
  assert(
    sym.rlevel == symbol_renaming_level::level1 ||
    sym.rlevel == symbol_renaming_level::level1_global);
  return symbol2tc(
    field->type,
    to_symbol2t(field).thename,
    sym.rlevel,
    sym.level1_num,
    0,
    sym.thread_num,
    0);
}

expr2tc split_structst::whole(const expr2tc &sym) const
{
  const symbol2t &s = to_symbol2t(sym);
  std::vector<expr2tc> members;
  for (const expr2tc &field : split.at(s.thename))
    members.push_back(at_level_of(field, s));
  return constant_struct2tc(sym->type, std::move(members));
}

/// An L2 name, or a member of one. An L2 name of a split object is the whole
/// read split_read assigned to its own symbol. A carried value holds it, and a
/// parameter assignment rewrites the renamed argument, so it must stay a read
/// of that definition.
static bool is_level2_read(const expr2tc &e)
{
  const expr2tc &base = is_member2t(e) ? to_member2t(e).source_value : e;
  if (!is_symbol2t(base))
    return false;
  const symbol_renaming_level level = to_symbol2t(base).rlevel;
  return level == symbol_renaming_level::level2 ||
         level == symbol_renaming_level::level2_global;
}

/// Whether rewrite leaves @p e as it is: nothing, an L2 read, or an address,
/// which names the object rather than its value.
static bool kept_whole(const expr2tc &e)
{
  return is_nil_expr(e) || is_level2_read(e) || is_address_of2t(e);
}

expr2tc split_structst::rewrite(const expr2tc &e) const
{
  if (kept_whole(e))
    return e;

  if (is_member2t(e) && is_symbol2t(to_member2t(e).source_value))
  {
    const member2t &m = to_member2t(e);
    const symbol2t &s = to_symbol2t(m.source_value);
    if (const std::vector<expr2tc> *f = fields(s.thename))
      return at_level_of((*f)[index.at(s.thename).at(m.member)], s);
    return e;
  }

  if (is_symbol2t(e))
    return fields(to_symbol2t(e).thename) ? whole(e) : e;

  // The eligibility scan keeps every byte-level access off a split object.
  assert(
    !(is_byte_extract2t(e) || is_byte_update2t(e) || is_bitcast2t(e)) || [&] {
      bool hit = false;
      e->foreach_operand([&](const expr2tc &op) {
        hit = hit || (is_symbol2t(op) && fields(to_symbol2t(op).thename));
      });
      return !hit;
    }());

  std::vector<expr2tc> ops;
  bool changed = false;
  e->foreach_operand([&](const expr2tc &op) {
    ops.push_back(rewrite(op));
    changed = changed || ops.back() != op;
  });
  if (!changed)
    return e;
  expr2tc r = e->clone();
  auto next = ops.begin();
  r->Foreach_operand([&](expr2tc &op) { op = *next++; });
  return r;
}
