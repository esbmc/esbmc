#include <goto-programs/scalarize_structs.h>
#include <irep2/irep2_utils.h>
#include <util/irep/pad_names.h>
#include <util/message/message.h>

namespace
{
/* The variables a scalarized struct is split into: one per member, nil for
 * padding, which nothing can read without the struct's address. */
struct fieldst
{
  std::vector<expr2tc> vars;
  std::unordered_map<irep_idt, size_t, irep_id_hash> index;
};

class scalarizert
{
public:
  scalarizert(goto_functionst &goto_functions, contextt &context)
    : goto_functions(goto_functions), context(context)
  {
  }

  void run()
  {
    for (auto &[name, f] : goto_functions.function_map)
      if (f.body_available)
        for (const auto &i : f.body.instructions)
          analyse(i);

    for (const auto &[name, type] : seen)
      if (!excluded.count(name))
        make_fields(name, type);
    if (fields.empty())
      return;

    for (auto &[name, f] : goto_functions.function_map)
      if (f.body_available)
        rewrite(f.body);
    goto_functions.update();
    log_status("Scalarized {} struct variable(s)", fields.size());
  }

private:
  goto_functionst &goto_functions;
  contextt &context;
  std::map<irep_idt, type2tc> seen;
  std::set<irep_idt> excluded;
  std::map<irep_idt, fieldst> fields;

  void exclude_all(const expr2tc &e)
  {
    if (is_nil_expr(e))
      return;
    if (is_symbol2t(e))
      excluded.insert(to_symbol2t(e).thename);
    e->foreach_operand([this](const expr2tc &op) { exclude_all(op); });
  }

  /* Excludes each symbol under @p e that is not the source of a member, where
   * @p e is written: a field variable can take a field write, but nothing can
   * stand for the whole struct there. */
  void exclude_whole(const expr2tc &e)
  {
    if (is_nil_expr(e))
      return;
    if (is_symbol2t(e))
    {
      excluded.insert(to_symbol2t(e).thename);
      return;
    }
    if (is_member2t(e) && is_symbol2t(to_member2t(e).source_value))
      return;
    e->foreach_operand([this](const expr2tc &op) { exclude_whole(op); });
  }

  void note(const expr2tc &e)
  {
    if (is_nil_expr(e))
      return;
    if (is_address_of2t(e))
      exclude_all(e);
    else if (is_symbol2t(e))
    {
      const irep_idt &name = to_symbol2t(e).thename;
      if (!is_struct_type(e->type))
        excluded.insert(name);
      else if (
        const auto [it, fresh] = seen.emplace(name, e->type);
        !fresh && it->second != e->type)
        excluded.insert(name);
    }
    e->foreach_operand([this](const expr2tc &op) { note(op); });
  }

  static bool mentions(const expr2tc &e, const irep_idt &name)
  {
    if (is_nil_expr(e))
      return false;
    if (is_symbol2t(e) && to_symbol2t(e).thename == name)
      return true;
    bool found = false;
    e->foreach_operand(
      [&](const expr2tc &op) { found = found || mentions(op, name); });
    return found;
  }

  void analyse(const goto_programt::instructiont &i)
  {
    note(i.code);
    note(i.guard);
    for (const expr2tc &inv : i.get_loop_invariants())
      note(inv);
    for (const expr2tc &target : i.get_loop_assigns_targets())
    {
      note(target);
      exclude_whole(target);
    }
    if (is_nil_expr(i.code))
      return;
    if (is_code_function_call2t(i.code))
      exclude_whole(to_code_function_call2t(i.code).ret);
    else if (is_code_assign2t(i.code))
    {
      const code_assign2t &a = to_code_assign2t(i.code);
      if (!is_symbol2t(a.target))
        exclude_whole(a.target);
      else if (
        a.source->type != a.target->type ||
        mentions(a.source, to_symbol2t(a.target).thename))
        excluded.insert(to_symbol2t(a.target).thename);
    }
    else if (is_code_decl2t(i.code))
    {
      const code_decl2t &d = to_code_decl2t(i.code);
      if (!is_nil_expr(d.init) && d.init->type != d.type)
        excluded.insert(d.value);
    }
  }

  /* A flexible or zero-length array member stays in its struct. */
  static bool layable(const type2tc &t)
  {
    if (!is_array_type(t))
      return true;
    const array_type2t &a = to_array_type(t);
    return !a.size_is_infinite && is_constant_int2t(a.array_size) &&
           !to_constant_int2t(a.array_size).value.is_zero();
  }

  void make_fields(const irep_idt &name, const type2tc &type)
  {
    const symbolt *s = context.find_symbol(name);
    if (!s || s->is_parameter || s->is_extern || !s->lvalue)
      return;
    const struct_type2t &st = to_struct_type(type);
    for (const type2tc &m : st.members)
      if (!layable(m))
        return;
    if (
      std::all_of(
        st.member_names.begin(), st.member_names.end(), [](irep_idt m) {
          return is_padding_name(id2string(m));
        }))
      return;

    fieldst &f = fields[name];
    const expr2tc &value = s->get_value2();
    for (size_t k = 0; k < st.members.size(); k++)
    {
      f.index.emplace(st.member_names[k], k);
      if (is_padding_name(id2string(st.member_names[k])))
      {
        f.vars.emplace_back();
        continue;
      }
      /* Named by index: passes such as k-induction's havoc classify a symbol
       * by substrings of its id, which a member name could add. */
      symbolt var = *s;
      var.id = id2string(name) + "::" + std::to_string(k);
      var.name =
        id2string(s->name) + "." + id2string(st.member_pretty_names[k]);
      var.set_type(st.members[k]);
      var.set_value(
        !is_nil_expr(value) && is_constant_struct2t(value)
          ? to_constant_struct2t(value).datatype_members[k]
          : expr2tc());
      f.vars.push_back(symbol2tc(st.members[k], var.id));
      context.add(var);
    }
  }

  bool scalarized(const expr2tc &e) const
  {
    return is_symbol2t(e) && fields.count(to_symbol2t(e).thename);
  }

  expr2tc whole(const expr2tc &sym) const
  {
    const fieldst &f = fields.at(to_symbol2t(sym).thename);
    const struct_type2t &st = to_struct_type(sym->type);
    std::vector<expr2tc> members;
    for (size_t k = 0; k < f.vars.size(); k++)
      members.push_back(
        is_nil_expr(f.vars[k]) ? gen_zero(st.members[k]) : f.vars[k]);
    return constant_struct2tc(sym->type, members);
  }

  expr2tc replace(const expr2tc &e) const
  {
    if (is_nil_expr(e))
      return e;
    if (is_member2t(e) && scalarized(to_member2t(e).source_value))
    {
      const member2t &m = to_member2t(e);
      const fieldst &f = fields.at(to_symbol2t(m.source_value).thename);
      return f.vars[f.index.at(m.member)];
    }
    if (scalarized(e))
      return whole(e);

    std::vector<expr2tc> ops;
    bool changed = false;
    e->foreach_operand([&](const expr2tc &op) {
      ops.push_back(replace(op));
      changed = changed || ops.back() != op;
    });
    if (!changed)
      return e;
    expr2tc r = e->clone();
    auto next = ops.begin();
    r->Foreach_operand([&](expr2tc &op) { op = *next++; });
    return r;
  }

  /* Member @p k of the struct value @p v, written into a field variable. */
  static expr2tc member_value(const expr2tc &v, size_t k)
  {
    if (is_constant_struct2t(v))
      return to_constant_struct2t(v).datatype_members[k];
    const struct_type2t &st = to_struct_type(v->type);
    if (
      is_sideeffect2t(v) &&
      to_sideeffect2t(v).kind == sideeffect2t::allockind::nondet)
      return sideeffect2tc(
        st.members[k],
        expr2tc(),
        expr2tc(),
        std::vector<expr2tc>(),
        type2tc(),
        sideeffect2t::allockind::nondet);
    return member2tc(st.members[k], v, st.member_names[k]);
  }

  /* The scalarized struct a DECL, DEAD or whole assignment @p code names. */
  irep_idt split_target(const expr2tc &code) const
  {
    irep_idt name;
    if (is_code_decl2t(code))
      name = to_code_decl2t(code).value;
    else if (is_code_dead2t(code))
      name = to_code_dead2t(code).value;
    else if (
      is_code_assign2t(code) && is_symbol2t(to_code_assign2t(code).target))
      name = to_symbol2t(to_code_assign2t(code).target).thename;
    return fields.count(name) ? name : irep_idt();
  }

  /* @p code, which names a scalarized struct, for member @p k's variable. */
  expr2tc for_field(const expr2tc &code, const expr2tc &value, size_t k) const
  {
    const expr2tc &var = fields.at(split_target(code)).vars[k];
    if (is_code_decl2t(code))
      return code_decl2tc(
        var->type,
        to_symbol2t(var).thename,
        is_nil_expr(value) ? expr2tc() : member_value(value, k),
        to_code_decl2t(code).location);
    if (is_code_dead2t(code))
      return code_dead2tc(
        var->type, to_symbol2t(var).thename, to_code_dead2t(code).location);
    return code_assign2tc(var, member_value(value, k));
  }

  void rewrite(goto_programt &body)
  {
    for (auto it = body.instructions.begin(); it != body.instructions.end();
         ++it)
    {
      it->guard = replace(it->guard);
      if (it->loop_contract_data)
      {
        for (expr2tc &inv : it->loop_contract_data->invariants)
          inv = replace(inv);
        for (expr2tc &target : it->loop_contract_data->assigns_targets)
          target = replace(target);
      }
      if (is_nil_expr(it->code))
        continue;
      const irep_idt name = split_target(it->code);
      if (name.empty())
      {
        it->code = replace(it->code);
        continue;
      }

      const expr2tc value = replace(
        is_code_decl2t(it->code)     ? to_code_decl2t(it->code).init
        : is_code_assign2t(it->code) ? to_code_assign2t(it->code).source
                                     : expr2tc());
      std::vector<expr2tc> codes;
      const std::vector<expr2tc> &vars = fields.at(name).vars;
      for (size_t k = 0; k < vars.size(); k++)
        if (!is_nil_expr(vars[k]))
          codes.push_back(for_field(it->code, value, k));

      /* The first takes this instruction's place, so jumps to it still land
       * on the split. */
      it->code = codes[0];
      for (size_t k = 1; k < codes.size(); k++)
      {
        it = body.instructions.insert(std::next(it), *it);
        it->code = codes[k];
        it->targets.clear();
        it->target_number = unsigned(-1);
        it->labels.clear();
      }
    }
  }
};
} // namespace

void scalarize_structs(goto_functionst &goto_functions, contextt &context)
{
  if (spawns_threads(goto_functions))
  {
    log_warning(
      "--scalarize-structs is ignored: the program may create "
      "threads");
    return;
  }
  scalarizert(goto_functions, context).run();
}
