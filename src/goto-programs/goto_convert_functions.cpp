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
  /* only "type" and "subtype", not "definition" */
  return id == exprt::i_type || id == typet::f_subtype;
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

    friend bool is_context(node_id v)
    {
      return v.type == CONTEXT;
    }

    friend bool is_symbol(node_id v)
    {
      return v.type == SYMBOL;
    }

    friend bool is_type(node_id v)
    {
      return v.type == TYPE;
    }

    friend bool is_expr(node_id v)
    {
      return v.type == EXPR;
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

  std::array<std::vector<nodet>, N_NODE_TYPES> Vs;

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
    /* Add root node */
    node_id v = add_node(&ctx);

    /* First put all symbol nodes into the graph and record a mapping from
     * symbol id to the node id. This will be used when collecting symbolic
     * types and therefore needs to be done before we start interpreting
     * expressions and types. */
    symbolst symbols;
    ctx.foreach_operand_in_order([this, v, &symbols](const symbolt &symbol) {
      node_id w = add_node(&symbol);
      symbols[symbol.id] = w;
      add_edge(v, {}, w);
    });

    /* Now that the map is filled, add expressions and types as nodes,
     * recursively. */
    for(const edge &e : adj(v))
    {
      node_id w = e.target;
      const symbolt *sym = (*this)[w].symbol;
      add_edge(w, "value", add_all(symbols, sym->value));
      add_edge(w, "type", add_all(symbols, sym->type));
    }

    return v;
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

  void collect(const symbolst &syms, node_id v, const irept &term, bool is_type)
  {
    /* Recurse through all the elements of the three component sets that make up
     * an irept:
     * - sub
     * - named_sub
     * - comments
     *
     * For each labelled element that denotes a (potentially thrashable) type,
     * record a new node in the graph and connect it to the given node 'v'.
     * In case the term is a "symbol" and we know it is in context of a type
     * (in contrast to a symbol-expression), look up the node corresponding to
     * the symbol and connect it, too.
     *
     * All other elements are not getting their own nodes since there is no
     * generic way of determining whether they are expressions or something
     * else.
     */

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
};

/* Map taking peculiarities of the graph into account, in particular the
 * indexing with node_id. */
template <typename T>
struct node_map
{
  std::array<std::vector<T>, context_type_grapht::N_NODE_TYPES> vecs;

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

void goto_convert_functionst::rename_types(
  typet &type,
  const symbolt &cur_name_sym)
{
  if(type.id() == typet::t_pointer)
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

  if(type.id() == typet::t_symbol)
  {
    /* unfold the definition of this symbol but do not recurse into it */
    type = type.definition();
    assert(!type.id().empty());
    return;
  }

  rename_exprs(type, cur_name_sym);
}

void goto_convert_functionst::rename_exprs(
  irept &expr,
  const symbolt &cur_name_sym)
{
  if(expr.id() == typet::t_pointer)
    return;

  Forall_irep(it, expr.get_sub())
    rename_exprs(*it, cur_name_sym);

  Forall_named_irep(it, expr.get_named_sub())
  {
    if(denotes_thrashable_subtype(it->first))
    {
      rename_types(static_cast<typet &>(it->second), cur_name_sym);
    }
    else
    {
      rename_exprs(it->second, cur_name_sym);
    }
  }

  Forall_named_irep(it, expr.get_comments())
    rename_exprs(it->second, cur_name_sym);
}

using node_id = context_type_grapht::node_id;
using edge = context_type_grapht::edge;

/* Support node_id hashing using types in std */
namespace std
{
template <>
struct hash<node_id> : hash<size_t>
{
  static_assert(sizeof(node_id) == sizeof(size_t));

  size_t operator()(const node_id &v) const noexcept
  {
    size_t w;
    memcpy(&w, &v, sizeof(w));
    // delegate to std lib for consistency; this is basically a no-op
    return hash<size_t>::operator()(w);
  }
};
} // namespace std

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

  std::function<void(const std::vector<node_id> &)> handle_scc;

  explicit sccst(
    const context_type_grapht &G,
    std::function<void(const std::vector<node_id> &)> handle_scc = {})
    : G(G), data(G), handle_scc(std::move(handle_scc))
  {
  }

  /* Tarjan's algorithm to compute SCCs.
   * See "Depth-first search and linear graph algorithms, SIAM, 1972.";
   * doi:10.1137/0201010; section 4 */
  void tarjan(node_id v)
  {
    size_t idx = ++index;
    data[v] = {idx, idx, true};
    stack.push_back(v);
    for(const edge &e : G.adj(v))
    {
      node_id w = e.target;
      size_t link;
      if(!data[w].index)
      {
        tarjan(w);
        link = data[w].lowlink;
      }
      else if(data[w].on_stack)
        link = data[w].index;
      else
        continue;
      data[v].lowlink = std::min(data[v].lowlink, link);
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

    if(handle_scc)
      handle_scc(scc);
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

  /* Tarjan's algorithm for computing strongly connected components (SCCs) has
   * the benefit of giving a topological ordering of the SCCs of the graph. We
   * use that as the order in which to replace/unfold symbolic types in symbols'
   * values and types. The effect is that these "dependencies" are resolved
   * before symbols that use on them. When a symbolic type refers to itself (or
   * to a another node in its SCC) it is "unfolded" exactly once.
   */

  context_type_grapht G;
  node_id root = G.add_all(context);

  G.forall_nodes(context_type_grapht::TYPE, [&G](node_id v) {
    const symbolt *sym = nullptr;
    for(edge e : G.adj(v))
      if(is_symbol(e.target) && e.label == typet::t_symbol)
      {
        sym = G[e.target].symbol;
        break;
      }
    if(!sym)
      return;
    typet *t = const_cast<typet *>(G[v].type);
    assert(t->id() == typet::t_symbol);
    t->definition() = sym->type; // copy
  });

  sccst(G, [this, &G](const std::vector<node_id> &scc) {
    for(node_id v : scc)
      if(is_symbol(v))
      {
        symbolt *sym = const_cast<symbolt *>(G[v].symbol);
        log_debug(
          "thrash-ts", "thrashing symbolic types inside symbol {}", sym->id);
        rename_exprs(sym->value, *sym);
        rename_types(sym->type, *sym);
      }
  }).tarjan(root);
}
