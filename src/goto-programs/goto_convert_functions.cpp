#include <cassert>
#include <goto-programs/goto_convert_functions.h>
#include <goto-programs/goto_inline.h>
#include <goto-programs/remove_no_op.h>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/i2string.h>
#include <util/prefix.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <util/type_byte_size.h>

goto_convert_functionst::goto_convert_functionst(
  contextt &_context,
  optionst &_options,
  goto_functionst &_functions)
  : goto_convertt(_context, _options), functions(_functions)
{
}

void goto_convert_functionst::goto_convert()
{
  // warning! hash-table iterators are not stable

  symbol_listt symbol_list;
  context.Foreach_operand_in_order([&symbol_list](symbolt &s) {
    if(!s.is_type && s.type.is_code())
      symbol_list.push_back(&s);
  });

  for(auto &it : symbol_list)
  {
    convert_function(*it);
  }

  functions.compute_location_numbers();
}

bool goto_convert_functionst::hide(const goto_programt &goto_program)
{
  for(const auto &instruction : goto_program.instructions)
  {
    for(const auto &label : instruction.labels)
    {
      if(label == "__ESBMC_HIDE")
        return true;
    }
  }

  return false;
}

void goto_convert_functionst::add_return(
  goto_functiont &f,
  const locationt &location)
{
  if(!f.body.instructions.empty() && f.body.instructions.back().is_return())
    return; // not needed, we have one already

  // see if we have an unconditional goto at the end
  if(
    !f.body.instructions.empty() && f.body.instructions.back().is_goto() &&
    is_true(f.body.instructions.back().guard))
    return;

  goto_programt::targett t = f.body.add_instruction();
  t->make_return();
  t->location = location;

  const typet &thetype = (f.type.return_type().id() == "symbol")
                           ? ns.follow(f.type.return_type())
                           : f.type.return_type();
  exprt rhs = exprt("sideeffect", thetype);
  rhs.statement("nondet");

  expr2tc tmp_expr;
  migrate_expr(rhs, tmp_expr);
  t->code = code_return2tc(tmp_expr);
}

void goto_convert_functionst::convert_function(symbolt &symbol)
{
  irep_idt identifier = symbol.id;

  // Apply a SFINAE test: discard unused C++ templates.
  // Note: can be removed probably? as the new clang-cpp-frontend should've
  // done a pretty good job at resolving template overloading
  if(
    symbol.value.get("#speculative_template") == "1" &&
    symbol.value.get("#template_in_use") != "1")
    return;

  // make tmp variables local to function
  tmp_symbol = symbol_generator(id2string(symbol.id) + "::$tmp::");

  auto it = functions.function_map.find(identifier);
  if(it == functions.function_map.end())
    functions.function_map.emplace(identifier, goto_functiont());

  goto_functiont &f = functions.function_map.at(identifier);
  f.type = to_code_type(symbol.type);
  f.body_available = symbol.value.is_not_nil();

  if(!f.body_available)
    return;

  if(!symbol.value.is_code())
  {
    log_error("got invalid code for function `{}'", id2string(identifier));
    abort();
  }

  const codet &code = to_code(symbol.value);

  locationt end_location;

  if(to_code(symbol.value).get_statement() == "block")
    end_location =
      static_cast<const locationt &>(to_code_block(code).end_location());
  else
    end_location.make_nil();

  // add "end of function"
  goto_programt tmp_end_function;
  goto_programt::targett end_function = tmp_end_function.add_instruction();
  end_function->type = END_FUNCTION;
  end_function->location = end_location;

  targets = targetst();
  targets.set_return(end_function);
  targets.has_return_value = f.type.return_type().id() != "empty" &&
                             f.type.return_type().id() != "constructor" &&
                             f.type.return_type().id() != "destructor";

  goto_convert_rec(code, f.body);

  // add non-det return value, if needed
  if(targets.has_return_value)
    add_return(f, end_location);

  // Wrap the body of functions name __VERIFIER_atomic_* with atomic_bengin
  // and atomic_end
  if(
    !f.body.instructions.empty() &&
    has_prefix(id2string(identifier), "c:@F@__VERIFIER_atomic_"))
  {
    goto_programt::instructiont a_begin;
    a_begin.make_atomic_begin();
    a_begin.location = f.body.instructions.front().location;
    f.body.insert_swap(f.body.instructions.begin(), a_begin);

    goto_programt::targett a_end = f.body.add_instruction();
    a_end->make_atomic_end();
    a_end->location = end_location;

    Forall_goto_program_instructions(i_it, f.body)
    {
      if(i_it->is_goto() && i_it->targets.front()->is_end_function())
      {
        i_it->targets.clear();
        i_it->targets.push_back(a_end);
      }
    }
  }

  // add "end of function"
  f.body.destructive_append(tmp_end_function);

  // do function tags (they are empty at this point)
  f.update_instructions_function(identifier);

  f.body.update();

  if(hide(f.body))
    f.body.hide = true;
}

void goto_convert(
  contextt &context,
  optionst &options,
  goto_functionst &functions)
{
  goto_convert_functionst goto_convert_functions(context, options, functions);

  goto_convert_functions.thrash_type_symbols();
  goto_convert_functions.goto_convert();
}

static bool denotes_thrashable_subtype(const irep_idt &id)
{
  return id == "type" || id == "subtype";
}

namespace
{
struct context_type_grapht
{
  enum node_typet : size_t
  {
    CONTEXT, /* root, top level */
    SYMBOL,  /* second level */
    TYPE,    /* higher level */
    EXPR,    /* higher level */
  };

  static constexpr size_t N_NODE_TYPES = EXPR + 1;

  template <typename T>
  static constexpr node_typet node_type(const T *)
  {
    if constexpr(std::is_same_v<T, contextt>)
      return CONTEXT;
    else if constexpr(std::is_same_v<T, symbolt>)
      return SYMBOL;
    else if constexpr(std::is_same_v<T, typet>)
      return TYPE;
    else
    {
      static_assert(std::is_same_v<T, exprt>);
      return EXPR;
    }
  }

  struct node_id
  {
    node_typet type : 2;
    size_t idx : CHAR_BIT * sizeof(size_t) - 2;

    friend bool operator==(const node_id &a, const node_id &b)
    {
      return a.type == b.type && a.idx == b.idx;
    }

    friend bool operator!=(const node_id &a, const node_id &b)
    {
      return !(a == b);
    }
  };

  struct edge
  {
    irep_idt label;
    node_id target;

    edge(irep_idt label, node_id target) : label(label), target(target)
    {
    }
  };

  struct nodet
  {
    union valuet
    {
      const contextt *context;
      const symbolt *symbol;
      const typet *type;
      const exprt *expr;
    } object;

    explicit nodet(const contextt *ctx)
    {
      object.context = ctx;
    }

    explicit nodet(const symbolt *sym)
    {
      object.symbol = sym;
    }

    explicit nodet(const typet *type)
    {
      object.type = type;
    }

    explicit nodet(const exprt *expr)
    {
      object.expr = expr;
    }

    std::vector<edge> adj;
  };

  std::vector<nodet> Vs[N_NODE_TYPES];

  template <typename F>
  void forall_nodes(node_typet type, F && f) const
  {
    for(size_t j = 0; j < Vs[type].size(); j++)
      f(node_id{type, j});
  }

  template <typename F>
  void forall_nodes(F &&f) const
  {
    for(size_t i = 0; i < N_NODE_TYPES; i++)
      forall_nodes((node_typet)i, f)
        ;
  }

  template <typename T>
  node_id add_node(const T *tgt)
  {
    node_typet t = node_type(tgt);
    node_id r = {t, Vs[t].size()};
    Vs[t].emplace_back(tgt);
    return r;
  }

  void add_edge(node_id v, irep_idt label, node_id w)
  {
    std::vector<edge> &adj = Vs[v.type][v.idx].adj;
    adj.emplace_back(label, w);
  }

  const nodet &node(node_id v) const
  {
    return Vs[v.type][v.idx];
  }

  const nodet::valuet &operator[](node_id v) const
  {
    return node(v).object;
  }

  const std::vector<edge> &adj(node_id v) const
  {
    return node(v).adj;
  }

  typedef std::unordered_map<irep_idt, node_id, irep_id_hash> symbolst;

  node_id add_all(const contextt &ctx)
  {
    node_id v = add_node(&ctx);
    symbolst symbols;
    ctx.foreach_operand_in_order([this, v, &symbols](const symbolt &symbol) {
      node_id w = add_node(&symbol);
      symbols[symbol.id] = w;
      add_edge(v, {}, w);
    });
    ctx.foreach_operand_in_order(
      [this, &symbols](const symbolt &symbol) { collect(symbols, symbol); });
    return v;
  }

  node_id collect(symbolst &syms, const symbolt &sym)
  {
    node_id v = syms.find(sym.id)->second;
    add_edge(v, "value", add_all(syms, sym.value));
    add_edge(v, "type", add_all(syms, sym.type));
    return v;
  }

  void collect(const symbolst &syms, node_id v, const irept &term, bool is_type)
  {
    forall_irep(it, term.get_sub())
      collect(syms, v, *it, false);
    forall_named_irep(it, term.get_named_sub())
    {
      if(denotes_thrashable_subtype(it->first))
        add_edge(
          v, it->first, add_all(syms, static_cast<const typet &>(it->second)));
      else if(
        is_type && term.id() == typet::t_symbol &&
        it->first == typet::a_identifier)
        add_edge(v, term.id(), syms.find(it->second.id())->second);
      else
        collect(syms, v, it->second, false);
    }
    forall_named_irep(it, term.get_comments())
    {
      if(denotes_thrashable_subtype(it->first))
        add_edge(
          v, it->first, add_all(syms, static_cast<const typet &>(it->second)));
      else
        collect(syms, v, it->second, false);
    }
  }

  node_id add_all(const symbolst &syms, const exprt &e)
  {
    node_id v = add_node(&e);
    collect(syms, v, e, false);
    return v;
  }

  node_id add_all(const symbolst &syms, const typet &t)
  {
    node_id v = add_node(&t);
    collect(syms, v, t, true);
    return v;
  }
};

template <typename T>
struct node_map
{
  std::array<std::vector<T>,context_type_grapht::N_NODE_TYPES> vecs;

  explicit node_map(const context_type_grapht &G, const T &init = {})
  {
    for(size_t i = 0; i < context_type_grapht::N_NODE_TYPES; i++)
      vecs[i].resize(G.Vs[i].size(), init);
  }

  T &operator[](context_type_grapht::node_id v)
  {
    return vecs[v.type][v.idx];
  }

  const T &operator[](context_type_grapht::node_id v) const
  {
    return vecs[v.type][v.idx];
  }
};
} // namespace

void goto_convert_functionst::collect_type(
  const irept &type,
  typename_sett &deps)
{
  if(type.id() == "pointer")
    return;

  if(type.id() == "symbol")
  {
    assert(type.identifier() != "");
    deps.insert(type.identifier());
    return;
  }

  collect_expr(type, deps);
}

void goto_convert_functionst::collect_expr(
  const irept &expr,
  typename_sett &deps)
{
  if(expr.id() == "pointer")
    return;

  forall_irep(it, expr.get_sub())
  {
    collect_expr(*it, deps);
  }

  forall_named_irep(it, expr.get_named_sub())
  {
    if(denotes_thrashable_subtype(it->first))
      collect_type(it->second, deps);
    else
      collect_expr(it->second, deps);
  }

  forall_named_irep(it, expr.get_comments())
  {
    if(denotes_thrashable_subtype(it->first))
      collect_type(it->second, deps);
    else
      collect_expr(it->second, deps);
  }
}

void goto_convert_functionst::rename_types(
  irept &type,
  const symbolt &cur_name_sym,
  const irep_idt &sname)
{
  if(type.id() == "pointer")
    return;

  // Some type symbols aren't entirely correct. This is because (in the current
  // 27_exStbFb test) some type symbols get the module name inserted into the
  // name -- so int32_t becomes main::int32_t.
  //
  // Now this makes entire sense, because int32_t could be something else in
  // some other file. However, because type symbols aren't squashed at type
  // checking time (which, you know, might make sense) we now don't know what
  // type symbol to link "int32_t" up to. So; instead we test to see whether
  // a type symbol is linked correctly, and if it isn't we look up what module
  // the current block of code came from and try to guess what type symbol it
  // should have.

  typet type2;
  if(type.id() == "symbol")
  {
    if(type.identifier() == sname)
    {
      // A recursive symbol -- the symbol we're about to link to is in fact the
      // one that initiated this chain of renames. This leads to either infinite
      // loops or segfaults, depending on the phase of the moon.
      // It should also never happen, but with C++ code it does, because methods
      // are part of the type, and methods can take a full struct/object as a
      // parameter, not just a reference/pointer. So, that's a legitimate place
      // where we have this recursive symbol dependancy situation.
      // The workaround to this is to just ignore it, and hope that it doesn't
      // become a problem in the future.
      return;
    }

    if(ns.lookup(type.identifier()))
    {
      // If we can just look up the current type symbol, use that.
      type2 = ns.follow((typet &)type);
    }
    else
    {
      // Otherwise, try to guess the namespaced type symbol
      std::string ident =
        cur_name_sym.module.as_string() + type.identifier().as_string();

      // Try looking that up.
      if(ns.lookup(irep_idt(ident)))
      {
        irept tmptype = type;
        tmptype.identifier(irep_idt(ident));
        type2 = ns.follow((typet &)tmptype);
      }
      else
      {
        // And if we fail
        log_error(
          "Can't resolve type symbol {} at symbol squashing time", ident);
        abort();
      }
    }

    type = type2;
    return;
  }

  rename_exprs(type, cur_name_sym, sname);
}

void goto_convert_functionst::rename_exprs(
  irept &expr,
  const symbolt &cur_name_sym,
  const irep_idt &sname)
{
  if(expr.id() == "pointer")
    return;

  Forall_irep(it, expr.get_sub())
    rename_exprs(*it, cur_name_sym, sname);

  Forall_named_irep(it, expr.get_named_sub())
  {
    if(denotes_thrashable_subtype(it->first))
    {
      rename_types(it->second, cur_name_sym, sname);
    }
    else
    {
      rename_exprs(it->second, cur_name_sym, sname);
    }
  }

  Forall_named_irep(it, expr.get_comments())
    rename_exprs(it->second, cur_name_sym, sname);
}

void goto_convert_functionst::wallop_type(
  irep_idt name,
  typename_mapt &typenames,
  const irep_idt &sname)
{
  // If this type doesn't depend on anything, no need to rename anything.
  typename_mapt::iterator it = typenames.find(name);
  assert(it != typenames.end());
  std::set<irep_idt> deps = std::exchange(it->second, {});
  if(deps.size() == 0)
    return;

  // Iterate over our dependancies ensuring they're resolved.
  for(const auto &dep : deps)
    wallop_type(dep, typenames, sname);

  // And finally perform renaming.
  symbolt *s = context.find_symbol(name);
  rename_types(s->type, *s, sname);
}

using node_id = context_type_grapht::node_id;
using edge = context_type_grapht::edge;

namespace
{
struct sccst /* strongly connected components */
{
  const context_type_grapht &G;

  struct tarjan_data
  {
    size_t index = 0;
    size_t lowlink;
    bool on_stack;
  };

  node_map<tarjan_data> data;
  std::vector<node_id> stack;
  size_t index = 0;

  std::unordered_set<irep_idt, irep_id_hash> large_scc_syms;

  sccst(const context_type_grapht &G) : G(G), data(G)
  {
  }

  void tarjan(node_id v) /* Tarjan's 1972 algorithm to compute SCCs */
  {
    size_t idx = ++index;
    data[v] = {idx, idx, true};
    stack.push_back(v);
    for(const edge &e : G.adj(v))
    {
      node_id w = e.target;
      if(!data[w].index)
      {
        tarjan(w);
        data[v].lowlink = std::min(data[v].lowlink, data[w].lowlink);
      }
      else if(data[w].on_stack)
      {
        data[v].lowlink = std::min(data[v].lowlink, data[w].index);
      }
    }

    if(data[v].lowlink != data[v].index)
      return;

    std::vector<node_id> scc;
    node_id w;
    do
    {
      w = stack.back();
      stack.pop_back();
      data[w].on_stack = false;
      scc.push_back(w);
    } while(v != w);

    // record SCCs of size > 1 to make life easier for thrash_type_symbols()
    if(scc.size() == 1)
      return;

    FILE *f = messaget::state.target("thrash-ts", VerbosityLevel::Debug);
    if(f)
    {
      fprintf(f, "scc:");
      for(node_id w : scc)
        fprintf(f, " %zd:%zd", w.type, w.idx);
      fprintf(f, " --");
    }
    for(node_id w : scc)
      if(w.type == context_type_grapht::SYMBOL)
      {
        const symbolt *sym = G[w].symbol;
        large_scc_syms.emplace(sym->id);
        if(f)
          fprintf(f, " %s", sym->id.c_str());
      }
    if(f)
      fprintf(f, "\n");
  }
};
} // namespace

void goto_convert_functionst::thrash_type_symbols()
{
  // This function has one purpose: remove as many type symbols as possible.
  // This is easy enough by just following each type symbol that occurs and
  // replacing it with the value of the type name. However, if we have a pointer
  // in a struct to itself, this breaks down. Therefore, don't rename types of
  // pointers; they have a type already; they're pointers.

  /* Some types under pointers we need to replace, however, in order to make
   * pointer arithmetic work on them.
   * See <https://github.com/esbmc/esbmc/issues/1289>. We do this by computing
   * the strongly connected components (SCCs) of the type subgraph of the graph
   * represented by the context. Those SCCs of size larger than one node are not
   * safe to be replaced at the moment since they would introduce cycles and our
   * ad-hoc recursions are not prepared for those...
   *
   * The original algorithm below this part is left intact, because it replaces
   * instances of large SCCs that do not occur under pointers. Eventually we
   * should remove it or replace it with a unified version. */

  context_type_grapht G;
  node_id root = G.add_all(context);
  sccst sccs(G);
  sccs.tarjan(root);
  const std::unordered_set<irep_idt, irep_id_hash> &avoid = sccs.large_scc_syms;
  G.forall_nodes(context_type_grapht::TYPE, [&G, &avoid](node_id v) {
    for(const edge &e : G.adj(v))
    {
      node_id w = e.target;
      if(e.label != "symbol")
        continue;
      assert(w.type == context_type_grapht::SYMBOL);
      const symbolt *sym = G[w].symbol;
      assert(sym->is_type);
      if(avoid.find(sym->id) != avoid.end())
        continue;
      *const_cast<typet *>(G[v].type) = sym->type;
    }
  });

  /* Original algorithm */

  // Collect a list of all type names. This is required before this entire
  // thing has no types, and there's no way (in C++ converted code at least)
  // to decide what name is a type or not.
  typename_sett names;
  context.foreach_operand([this, &names](const symbolt &s) {
    collect_expr(s.value, names);
    collect_type(s.type, names);
  });

  // Try to compute their dependencies.

  typename_mapt typenames;
  context.foreach_operand([this, &names, &typenames](const symbolt &s) {
    if(names.find(s.id) != names.end())
    {
      typename_sett list;
      collect_expr(s.value, list);
      collect_type(s.type, list);
      typenames[s.id] = list;
    }
  });

  for(auto &it : typenames)
    it.second.erase(it.first);

  // Now, repeatedly rename all types. When we encounter a type that contains
  // unresolved symbols, resolve it first, then include it into this type.
  // This means that we recurse to whatever depth of nested types the user
  // has. With at least a meg of stack, I doubt that's really a problem.
  std::map<irep_idt, std::set<irep_idt>>::iterator it;
  for(it = typenames.begin(); it != typenames.end(); it++)
    wallop_type(it->first, typenames, it->first);

  // And now all the types have a fixed form, rename types in all existing code.
  context.Foreach_operand([this](symbolt &s) {
    rename_types(s.type, s, s.id);
    rename_exprs(s.value, s, s.id);
  });
}
