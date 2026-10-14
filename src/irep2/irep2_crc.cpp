#include <array>
#include <tuple>
#include <type_traits>
#include <vector>
#include <irep2/irep2_dispatch.h>

namespace
{
struct irep2_node_ref
{
  const expr2t *expr;
  const type2t *type;
};

struct irep2_traversal_frame
{
  irep2_node_ref node;
  bool expanded;
};

template <class T>
size_t do_type_crc(const T &field)
{
  if constexpr (std::is_enum_v<T>)
    return std::hash<uint8_t>{}(static_cast<uint8_t>(field));
  else
    return std::hash<T>{}(field);
}

template <typename Sink>
void feed_bigint(const BigInt &value, Sink &&sink)
{
  const uint8_t sign = value.is_positive() ? 1 : 0;
  sink(&sign, sizeof(sign));

  if (value.is_zero())
    return;

  std::array<unsigned char, 256> stack_buf;
  if (value.dump(stack_buf.data(), stack_buf.size()))
  {
    sink(stack_buf.data(), stack_buf.size());
    return;
  }

  std::vector<unsigned char> heap_buf(stack_buf.size() * 2);
  while (!value.dump(heap_buf.data(), heap_buf.size()))
    heap_buf.resize(heap_buf.size() * 2);
  sink(heap_buf.data(), heap_buf.size());
}

size_t do_type_crc(const BigInt &value)
{
  size_t crc = 0;
  feed_bigint(value, [&](const unsigned char *data, size_t len) {
    for (size_t i = 0; i < len; ++i)
      esbmct::hash_combine(crc, data[i]);
  });
  return crc;
}

size_t do_type_crc(const fixedbvt &value)
{
  return do_type_crc(BigInt(value.to_ansi_c_string().c_str()));
}

size_t do_type_crc(const ieee_floatt &value)
{
  return do_type_crc(value.pack());
}

size_t do_type_crc(const std::vector<irep_idt> &values)
{
  size_t crc = 0;
  for (const irep_idt &value : values)
    esbmct::hash_combine(crc, value.hash());
  return crc;
}

size_t do_type_crc(const irep_idt &value)
{
  return value.hash();
}

template <typename F>
void enumerate_irep2_children(const expr2tc &child, F &f)
{
  if (const expr2t *ptr = child.get())
    f(irep2_node_ref{ptr, nullptr});
}

template <typename F>
void enumerate_irep2_children(const type2tc &child, F &f)
{
  if (const type2t *ptr = child.get())
    f(irep2_node_ref{nullptr, ptr});
}

template <typename F>
void enumerate_irep2_children(const std::vector<expr2tc> &children, F &f)
{
  for (const expr2tc &child : children)
    enumerate_irep2_children(child, f);
}

template <typename F>
void enumerate_irep2_children(const std::vector<type2tc> &children, F &f)
{
  for (const type2tc &child : children)
    enumerate_irep2_children(child, f);
}

template <typename T, typename F>
void enumerate_irep2_children(const T &, F &)
{
}

template <class K, typename F>
void enumerate_node_children(const K &node, F &f)
{
  std::apply(
    [&](auto... mp) { (enumerate_irep2_children(node.*mp, f), ...); },
    K::fields);
}

size_t cached_crc(const irep2_node_ref &node)
{
  if (node.expr != nullptr)
    return node.expr->crc_val.load(std::memory_order_acquire);
  return node.type->crc_val.load(std::memory_order_acquire);
}

template <class T>
size_t do_cached_child_crc(const T &field)
{
  return do_type_crc(field);
}

size_t do_cached_child_crc(const expr2tc &field)
{
  if (const expr2t *ptr = field.get())
    return ptr->crc_val.load(std::memory_order_acquire);
  return std::hash<uint8_t>{}(0);
}

size_t do_cached_child_crc(const type2tc &field)
{
  if (const type2t *ptr = field.get())
    return ptr->crc_val.load(std::memory_order_acquire);
  return std::hash<uint8_t>{}(0);
}

size_t do_cached_child_crc(const std::vector<expr2tc> &field)
{
  size_t crc = 0;
  for (const expr2tc &child : field)
    esbmct::hash_combine(crc, do_cached_child_crc(child));
  return crc;
}

size_t do_cached_child_crc(const std::vector<type2tc> &field)
{
  size_t crc = 0;
  for (const type2tc &child : field)
    esbmct::hash_combine(crc, do_cached_child_crc(child));
  return crc;
}

template <class K>
size_t compute_crc_from_cached_children(const K &node)
{
  if (size_t cached = node.crc_val.load(std::memory_order_acquire); cached != 0)
    return cached;

  size_t crc = 0;
  if constexpr (std::is_base_of_v<expr2t, K>)
    esbmct::hash_combine(crc, do_type_crc(node.expr_id));
  else
    esbmct::hash_combine(crc, do_type_crc(node.type_id));

  std::apply(
    [&](auto... mp) {
      (esbmct::hash_combine(crc, do_cached_child_crc(node.*mp)), ...);
    },
    K::fields);
  node.crc_val.store(crc, std::memory_order_release);
  return crc;
}

template <typename F>
void foreach_irep2_child(const irep2_node_ref &node, F &f)
{
  if (node.expr != nullptr)
  {
    switch (node.expr->expr_id)
    {
#define IREP2_EXPR(kind, _)                                                    \
  case expr2t::kind##_id:                                                      \
    enumerate_node_children(static_cast<const kind##2t &>(*node.expr), f);     \
    return;
#include <irep2/expr_kinds.inc>
#undef IREP2_EXPR
    case expr2t::end_expr_id:
      break;
    }
    std::unreachable();
  }

  switch (node.type->type_id)
  {
#define IREP2_TYPE(kind, _)                                                    \
  case type2t::kind##_id:                                                      \
    enumerate_node_children(                                                   \
      static_cast<const kind##_type2t &>(*node.type), f);                      \
    return;
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE
  case type2t::end_type_id:
    break;
  }
  std::unreachable();
}

void compute_crc_from_cached_children(const irep2_node_ref &node)
{
  if (node.expr != nullptr)
  {
    switch (node.expr->expr_id)
    {
#define IREP2_EXPR(kind, _)                                                    \
  case expr2t::kind##_id:                                                      \
    compute_crc_from_cached_children(                                          \
      static_cast<const kind##2t &>(*node.expr));                              \
    return;
#include <irep2/expr_kinds.inc>
#undef IREP2_EXPR
    case expr2t::end_expr_id:
      break;
    }
    std::unreachable();
  }

  switch (node.type->type_id)
  {
#define IREP2_TYPE(kind, _)                                                    \
  case type2t::kind##_id:                                                      \
    compute_crc_from_cached_children(                                          \
      static_cast<const kind##_type2t &>(*node.type));                         \
    return;
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE
  case type2t::end_type_id:
    break;
  }
  std::unreachable();
}

size_t iterative_crc(const irep2_node_ref &root)
{
  if (size_t cached = cached_crc(root); cached != 0)
    return cached;

  std::vector<irep2_traversal_frame> stack{{root, false}};
  while (!stack.empty())
  {
    const irep2_traversal_frame frame = stack.back();
    stack.pop_back();

    if (cached_crc(frame.node) != 0)
      continue;

    if (frame.expanded)
    {
      compute_crc_from_cached_children(frame.node);
      continue;
    }

    stack.push_back({frame.node, true});
    auto push_uncached_child = [&](const irep2_node_ref &child) {
      if (cached_crc(child) == 0)
        stack.push_back({child, false});
    };
    foreach_irep2_child(frame.node, push_uncached_child);
  }

  return cached_crc(root);
}
} // namespace

size_t expr2t::crc() const
{
  return iterative_crc(irep2_node_ref{this, nullptr});
}

size_t type2t::crc() const
{
  return iterative_crc(irep2_node_ref{nullptr, this});
}
