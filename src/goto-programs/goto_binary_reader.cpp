#include <goto-programs/goto_binary_reader.h>
#include <goto-programs/read_bin_goto_object.h>
#include <goto-programs/read_cbmc_goto_object.h>
#include <goto-programs/goto_functions.h>
#include <util/message.h>
#include <fstream>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>

bool goto_binary_reader::read_goto_binary_array(
  const void *data,
  size_t size,
  contextt &context,
  contextt &ignored,
  goto_functionst &dest)
{
  using namespace boost::iostreams;
  stream<array_source> src(static_cast<const char *>(data), size);
  return read_bin_goto_object(src, "", context, ignored, function_set, dest);
}

bool goto_binary_reader::read_goto_binary(
  const std::string &path,
  contextt &context,
  goto_functionst &dest)
{
  std::ifstream in(path, std::ios::in | std::ios::binary);

  // Auto-detect the format from the magic header: a CBMC goto-binary starts
  // with 0x7f 'G' 'B' 'F', whereas ESBMC's own format starts with 'G' 'B' 'F'.
  char hdr[4] = {0, 0, 0, 0};
  in.read(hdr, 4);
  std::streamsize got = in.gcount();
  in.clear();
  in.seekg(0, std::ios::beg);

  const unsigned char *uhdr = reinterpret_cast<const unsigned char *>(hdr);
  if (got >= 4 && is_cbmc_goto_magic(uhdr))
    return read_cbmc_goto_object(in, path, context, dest);

  return read_bin_goto_object(in, path, context, dest);
}
