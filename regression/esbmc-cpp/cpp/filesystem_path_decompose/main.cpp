#include <filesystem>
#include <cassert>

namespace fs = std::filesystem;

int main()
{
  fs::path p("/a/b.txt");
  assert(p.filename() == fs::path("b.txt"));
  assert(p.parent_path() == fs::path("/a"));
  assert(p.extension() == fs::path(".txt"));
  assert(p.stem() == fs::path("b"));
  assert(p.has_filename());
  assert(p.has_extension());

  // [fs.path.decompose]/7
  assert(fs::path("/.rc").extension() == fs::path(""));
  assert(fs::path("/.rc").stem() == fs::path(".rc"));
  assert(!fs::path("/.rc").has_extension());
  assert(fs::path("/a/..").extension() == fs::path(""));
  assert(fs::path("/a/..").stem() == fs::path(".."));

  // A trailing period is an extension; only the last one splits.
  assert(fs::path("b.").extension() == fs::path("."));
  assert(fs::path("a.b.c").stem() == fs::path("a.b"));

  // A separator run collapses, except when it is the root itself.
  assert(fs::path("a//b").parent_path() == fs::path("a"));
  assert(fs::path("//b").parent_path() == fs::path("//"));
  assert(fs::path("/x").parent_path() == fs::path("/"));

  fs::path r("q");
  assert(r.filename() == fs::path("q"));
  assert(r.parent_path() == fs::path(""));
  return 0;
}
