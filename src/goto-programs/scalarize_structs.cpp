#include <goto-programs/scalarize_structs.h>
#include <irep2/irep2_utils.h>
#include <util/irep/pad_names.h>
#include <util/message/message.h>

namespace
{
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
  std::map<irep_idt, std::vector<expr2tc>> fields;

  void exclude_all(const expr2tc &e)
  {
    if (is_nil_expr(e))
      return;
    if (is_symbol2t(e))
      excluded.insert(to_symbol2t(e).thename);
    e->foreach_operand([this](const expr2tc &op) { exclude_all(op); });
  }

  /* Excludes each symbol under @p e that is not the source of a member, where
   * @p e is written: a field symbol can take a field write, but nothing can
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
    else if (is_symbol2t(e) && is_struct_type(e->type))
    {
      const auto [it, fresh] = seen.emplace(to_symbol2t(e).thename, e->type);
      if (!fresh && it->second != e->type)
        excluded.insert(it->first);
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
    if (is_nil_expr(i.code))
      return;
    if (is_code_function_call2t(i.code))
      exclude_whole(to_code_function_call2t(i.code).ret);
    else if (is_code_assign2t(i.code))
    {
      const code_assign2t &a = to_code_assign2t(i.code);
      if (!is_symbol2t(a.target))
        exclude_whole(a.target);
      else if (mentions(a.source, to_symbol2t(a.target).thename))
        excluded.insert(to_symbol2t(a.target).thename);
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
    if (st.members.empty())
      return;
    for (const type2tc &m : st.members)
      if (!layable(m))
        return;

    std::vector<symbolt> news;
    for (size_t k = 0; k < st.members.size(); k++)
    {
      symbolt f = *s;
      /* Symex leaves a symbol named with '$' out of traces, and padding has
       * no value to show. Not '.': a value-set entry is its symbol's name
       * followed by a suffix that starts with '.' or '[', so "s.f" + ".g"
       * would read as "s" + ".f.g". */
      const std::string &m = id2string(st.member_names[k]);
      f.id = id2string(name) + (is_padding_name(m) ? "$" : "::") + m;
      f.name = id2string(s->name) + "." + id2string(st.member_pretty_names[k]);
      if (context.find_symbol(f.id))
        return;
      f.set_type(st.members[k]);
      const expr2tc &v = s->get_value2();
      f.set_value(
        !is_nil_expr(v) && is_constant_struct2t(v)
          ? to_constant_struct2t(v).datatype_members[k]
          : expr2tc());
      news.push_back(f);
    }

    std::vector<expr2tc> &syms = fields[name];
    for (symbolt &f : news)
    {
      syms.push_back(symbol2tc(f.get_type2(), f.id));
      context.add(f);
    }
  }

  expr2tc field(const expr2tc &whole, const irep_idt &member) const
  {
    const struct_type2t &st = to_struct_type(whole->type);
    const std::vector<expr2tc> &syms = fields.at(to_symbol2t(whole).thename);
    for (size_t k = 0; k < st.member_names.size(); k++)
      if (st.member_names[k] == member)
        return syms[k];
    log_error("scalarize_structs: no member {}", member);
    abort();
  }

  bool scalarized(const expr2tc &e) const
  {
    return is_symbol2t(e) && fields.count(to_symbol2t(e).thename);
  }

  expr2tc replace(const expr2tc &e) const
  {
    if (is_nil_expr(e))
      return e;
    if (is_member2t(e) && scalarized(to_member2t(e).source_value))
      return field(to_member2t(e).source_value, to_member2t(e).member);
    if (scalarized(e))
      return constant_struct2tc(e->type, fields.at(to_symbol2t(e).thename));

    expr2tc r = e->clone();
    bool changed = false;
    r->Foreach_operand([&](expr2tc &op) {
      expr2tc n = replace(op);
      if (n != op)
      {
        op = n;
        changed = true;
      }
    });
    return changed ? r : e;
  }

  /* Member @p k of the struct value @p v, written into a field symbol. */
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

  /* One instruction per field in place of @p it, which keeps its position so
   * jumps to it still land on the first. */
  static void expand(
    goto_programt &body,
    goto_programt::targett it,
    const std::vector<expr2tc> &codes)
  {
    it->code = codes[0];
    auto pos = std::next(it);
    for (size_t k = 1; k < codes.size(); k++)
    {
      auto t = body.instructions.insert(pos, *it);
      t->code = codes[k];
      t->targets.clear();
      t->target_number = unsigned(-1);
      t->labels.clear();
    }
  }

  void rewrite(goto_programt &body)
  {
    for (auto it = body.instructions.begin(); it != body.instructions.end();
         ++it)
    {
      it->guard = replace(it->guard);
      if (is_nil_expr(it->code))
        continue;
      std::vector<expr2tc> codes;
      if (
        is_code_decl2t(it->code) &&
        fields.count(to_code_decl2t(it->code).value))
      {
        const code_decl2t &d = to_code_decl2t(it->code);
        const expr2tc init = replace(d.init);
        const auto &syms = fields.at(d.value);
        for (size_t k = 0; k < syms.size(); k++)
          codes.push_back(code_decl2tc(
            syms[k]->type,
            to_symbol2t(syms[k]).thename,
            is_nil_expr(init) ? expr2tc() : member_value(init, k),
            d.location));
      }
      else if (
        is_code_dead2t(it->code) &&
        fields.count(to_code_dead2t(it->code).value))
      {
        const code_dead2t &d = to_code_dead2t(it->code);
        for (const expr2tc &f : fields.at(d.value))
          codes.push_back(
            code_dead2tc(f->type, to_symbol2t(f).thename, d.location));
      }
      else if (
        is_code_assign2t(it->code) &&
        scalarized(to_code_assign2t(it->code).target))
      {
        const code_assign2t &a = to_code_assign2t(it->code);
        const expr2tc source = replace(a.source);
        const auto &syms = fields.at(to_symbol2t(a.target).thename);
        for (size_t k = 0; k < syms.size(); k++)
          codes.push_back(code_assign2tc(syms[k], member_value(source, k)));
      }
      else
      {
        it->code = replace(it->code);
        continue;
      }
      expand(body, it, codes);
      std::advance(it, codes.size() - 1);
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
