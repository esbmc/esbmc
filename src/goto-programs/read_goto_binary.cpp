#include <goto-programs/read_bin_goto_object.h>
#include <goto-programs/read_goto_binary.h>
#include <fstream>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>

bool read_goto_binary_array(
  const void *data,
  size_t size,
  contextt &context,
  goto_functionst &dest)
{
  using namespace boost::iostreams;
  stream<array_source> src(static_cast<const char *>(data), size);
  return read_bin_goto_object(src, "", context, dest);
}

bool read_goto_binary(
  const std::string &path,
  contextt &context,
  goto_functionst &dest)
{
  std::ifstream in(path, std::ios::in | std::ios::binary);
  return read_bin_goto_object(in, path, context, dest);
}
