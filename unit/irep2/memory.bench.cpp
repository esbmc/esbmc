#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/config.h>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <new>
#include <string>
#include <sstream>

// ============================================================================
// Allocation accounting via global operator new/delete overrides.
//
// Counters are toggled on by `alloc_stats::enabled = true` so test cases
// not interested in counting (e.g. Catch2 microbenches that need clean
// timings) can leave them off and pay zero overhead beyond the branch.
// ============================================================================
namespace alloc_stats
{
std::atomic<bool> enabled{false};
std::atomic<std::size_t> count{0};
std::atomic<std::size_t> bytes{0};

struct snapshot
{
  std::size_t count;
  std::size_t bytes;
};

snapshot take()
{
  return {
    count.load(std::memory_order_relaxed),
    bytes.load(std::memory_order_relaxed)};
}

struct delta
{
  std::size_t count;
  std::size_t bytes;
};

delta diff(const snapshot &before, const snapshot &after)
{
  return {after.count - before.count, after.bytes - before.bytes};
}

// RAII window: turns counting on for the lifetime of the object,
// captures the entry snapshot, and exposes the delta on demand.
struct window
{
  snapshot entry;
  bool prev_enabled;
  window() : entry(take()), prev_enabled(enabled.exchange(true))
  {
  }
  ~window()
  {
    enabled.store(prev_enabled, std::memory_order_relaxed);
  }
  delta now() const
  {
    return diff(entry, take());
  }
};
} // namespace alloc_stats

// Global new/delete: a couple of cycles of overhead on the common path
// when disabled (atomic load + branch). We avoid replacing the array
// forms; they're rarer and not the irep2 allocation path.
//
// Pair with the standard `::operator new` / `::operator delete` rather
// than malloc/free so the compiler's allocation-tracking
// (-Wmismatched-new-delete) sees consistent allocator usage across
// inlined call sites.
void *operator new(std::size_t sz)
{
  void *p = ::operator new(sz, std::nothrow);
  if (!p)
    throw std::bad_alloc{};
  if (alloc_stats::enabled.load(std::memory_order_relaxed))
  {
    alloc_stats::count.fetch_add(1, std::memory_order_relaxed);
    alloc_stats::bytes.fetch_add(sz, std::memory_order_relaxed);
  }
  return p;
}

void operator delete(void *p) noexcept
{
  ::operator delete(p, std::nothrow);
}

void operator delete(void *p, std::size_t /*sz*/) noexcept
{
  ::operator delete(p, std::nothrow);
}

// ============================================================================
// /proc/self/status RSS helper. Linux-only; the existing test infra is
// Linux-only too, so no need to abstract.
// ============================================================================
namespace rss
{
// Returns the named /proc/self/status field in kB, or 0 if not found.
std::size_t field_kb(const char *name)
{
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
  {
    if (line.rfind(name, 0) != 0)
      continue;
    std::istringstream is(line);
    std::string tag;
    std::size_t value = 0;
    is >> tag >> value;
    return value;
  }
  return 0;
}

std::size_t vm_rss_kb()
{
  return field_kb("VmRSS:");
}
std::size_t vm_peak_kb()
{
  return field_kb("VmPeak:");
}
} // namespace rss

// ============================================================================
// Synthesis helpers (same conventions as irep2.bench.cpp).
// ============================================================================
namespace
{
type2tc word_type()
{
  return get_uint_type(config.ansi_c.word_size);
}

expr2tc gen_const(unsigned v)
{
  return constant_int2tc(word_type(), BigInt(v));
}

expr2tc gen_symbol(const std::string &name)
{
  return symbol2tc(word_type(), irep_idt(name));
}

// Build N symbol-vs-constant equality predicates. Each is a small node
// tree (symbol2t + constant_int2t + equality2t) — three irep2 allocs.
// Useful baseline for bulk-allocation pressure tests.
std::vector<expr2tc> gen_equalities(unsigned n)
{
  std::vector<expr2tc> v;
  v.reserve(n);
  for (unsigned i = 0; i < n; ++i)
    v.push_back(equality2tc(gen_symbol("v" + std::to_string(i)), gen_const(i)));
  return v;
}
} // namespace

// ============================================================================
// 1. Per-op latency. Pure Catch2 microbenches; alloc counting OFF.
// ============================================================================

TEST_CASE("memory: per-op latency", "[bench]")
{
  config.ansi_c.word_size = 32;

  BENCHMARK("constant_int2tc(...)")
  {
    return gen_const(42);
  };
  BENCHMARK("symbol2tc(...)")
  {
    return gen_symbol("v");
  };
  BENCHMARK("equality2tc(sym, const)")
  {
    return equality2tc(gen_symbol("v"), gen_const(0));
  };
  BENCHMARK("constant_int copy x 1")
  {
    expr2tc base = gen_const(1);
    expr2tc copy = base;
    return copy;
  };
  BENCHMARK("constant_int copy x 8")
  {
    expr2tc base = gen_const(1);
    expr2tc c0 = base, c1 = base, c2 = base, c3 = base;
    expr2tc c4 = base, c5 = base, c6 = base, c7 = base;
    return c0;
  };
}

// ============================================================================
// 2. Allocation-count workloads. Catch2 INFO/WARN drives the reporting,
//    no BENCHMARK macro because the counter overhead would dominate
//    timings.
// ============================================================================

TEST_CASE("memory: bulk allocation pressure", "[probe]")
{
  config.ansi_c.word_size = 32;

  const unsigned counts[] = {64, 1024, 10000};
  for (unsigned n : counts)
  {
    alloc_stats::window w;
    {
      auto v = gen_equalities(n);
      static_cast<void>(v);
    }
    auto d = w.now();
    WARN(
      "build+destroy " << n << " equality nodes: " << d.count << " allocs, "
                       << d.bytes << " bytes total");
  }
}

TEST_CASE("memory: repeated-name symbol creation", "[probe]")
{
  config.ansi_c.word_size = 32;

  // Symbols whose names repeat should ideally share storage for the
  // irep_idt payload. Construct 10000 fresh symbols using only 16
  // distinct names; allocations beyond the unique name set indicate
  // missed interning opportunities.
  const unsigned total = 10000;
  const unsigned unique = 16;

  std::vector<std::string> names;
  names.reserve(unique);
  for (unsigned i = 0; i < unique; ++i)
    names.push_back("v" + std::to_string(i));

  alloc_stats::window w;
  std::vector<expr2tc> nodes;
  nodes.reserve(total);
  for (unsigned i = 0; i < total; ++i)
    nodes.push_back(gen_symbol(names[i % unique]));
  auto d = w.now();
  WARN(
    "10000 symbol nodes from 16 unique names: " << d.count << " allocs, "
                                                << d.bytes << " bytes");
}

TEST_CASE("memory: small-kind allocation hammer", "[probe]")
{
  config.ansi_c.word_size = 32;

  // What share of total allocations comes from the small fixed-size
  // expr kinds? Build 50000 mixed small nodes.
  const unsigned n = 50000;

  alloc_stats::window w;
  std::vector<expr2tc> nodes;
  nodes.reserve(n);
  for (unsigned i = 0; i < n; ++i)
  {
    switch (i % 3)
    {
    case 0:
      nodes.push_back(gen_const(i));
      break;
    case 1:
      nodes.push_back(gen_symbol("s" + std::to_string(i)));
      break;
    case 2:
      nodes.push_back(equality2tc(gen_symbol("e"), gen_const(i)));
      break;
    }
  }
  auto d = w.now();
  WARN(
    "50000 mixed small-kind nodes: " << d.count << " allocs, " << d.bytes
                                     << " bytes");
}

TEST_CASE("memory: irep_container copy storm", "[probe]")
{
  config.ansi_c.word_size = 32;

  // One root node, copied N times. Refcount-only copies should be
  // alloc-free; any allocations here are vector growth in the holder.
  expr2tc root = equality2tc(gen_symbol("v"), gen_const(0));

  alloc_stats::window w;
  std::vector<expr2tc> copies;
  copies.reserve(100000);
  for (unsigned i = 0; i < 100000; ++i)
    copies.push_back(root);
  auto d = w.now();
  WARN(
    "100000 refcount copies of one node: " << d.count << " allocs, " << d.bytes
                                           << " bytes");
}

// ============================================================================
// 3. RSS / peak-bytes proxy. A synthetic symex-shaped workload: build
//    a tree of N×M assignments, hold it live, then capture the RSS
//    delta. This is the closest microbench proxy for "would the arena
//    actually reduce peak memory".
// ============================================================================

TEST_CASE("memory: synthetic symex workload RSS", "[probe]")
{
  config.ansi_c.word_size = 32;

  // Force a couple of allocations *before* the snapshot so the
  // process has its steady-state allocator footprint already in
  // place — without this, the first measurement absorbs initial
  // heap growth that's unrelated to the workload.
  {
    std::vector<expr2tc> warmup(1024);
  }

  const std::size_t rss_before = rss::vm_rss_kb();
  const std::size_t peak_before = rss::vm_peak_kb();

  // 50000 variables, each appearing in 4 equality predicates with
  // different constants. ~200000 small expr2tc nodes held live.
  // Sized to cross the page-granularity floor (/proc/self/status
  // reports in kB, and the holder vector + heap fragments need to
  // be large enough to show up).
  std::vector<expr2tc> live;
  live.reserve(200000);
  for (unsigned v = 0; v < 50000; ++v)
  {
    expr2tc sym = gen_symbol("x" + std::to_string(v));
    for (unsigned c = 0; c < 4; ++c)
      live.push_back(equality2tc(sym, gen_const(c)));
  }

  const std::size_t rss_after = rss::vm_rss_kb();
  const std::size_t peak_after = rss::vm_peak_kb();

  WARN(
    "synthetic workload: VmRSS delta = "
    << (rss_after - rss_before) << " kB, VmPeak delta = "
    << (peak_after - peak_before) << " kB (live nodes: " << live.size() << ")");

  // Hold `live` to end of scope so the after-snapshot is meaningful.
  REQUIRE(live.size() == 200000);
}
