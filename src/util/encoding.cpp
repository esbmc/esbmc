#include <util/encoding.h>
#include <stdexcept>

std::vector<uint8_t> base64_decode(const std::string &encoded_string)
{
  static constexpr unsigned char kDecodingTable[] = {
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, // 0-15
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, // 16-31
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 62, 64, 64, 64, 63, // 32-47
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 64, 64, 0,  64, 64, // 48-63
    64, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, // 64-79
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 64, 64, 64, 64, // 80-95
    64, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, // 96-111
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 64, 64, 64, 64, 64  // 112-127
  };

  size_t in_len = encoded_string.size();
  if (in_len % 4 != 0)
    throw std::runtime_error("Input data size is not a multiple of 4");

  size_t padding = 0;
  if (in_len > 0 && (encoded_string[in_len - 1] == '='))
    padding++;
  if (in_len > 1 && (encoded_string[in_len - 2] == '='))
    padding++;

  size_t out_len = (in_len * 3) / 4 - padding;
  std::vector<uint8_t> decoded(out_len);

  for (size_t i = 0, j = 0; i < in_len;)
  {
    uint32_t a =
      encoded_string[i] == '='
        ? 0 & i++
        : kDecodingTable[static_cast<unsigned char>(encoded_string[i++])];
    uint32_t b =
      encoded_string[i] == '='
        ? 0 & i++
        : kDecodingTable[static_cast<unsigned char>(encoded_string[i++])];
    uint32_t c =
      encoded_string[i] == '='
        ? 0 & i++
        : kDecodingTable[static_cast<unsigned char>(encoded_string[i++])];
    uint32_t d =
      encoded_string[i] == '='
        ? 0 & i++
        : kDecodingTable[static_cast<unsigned char>(encoded_string[i++])];

    uint32_t triple = (a << 18) | (b << 12) | (c << 6) | d;

    for (int k = 2; k >= 0 && j < out_len; --k)
      decoded[j++] = (triple >> (8 * k)) & 0xFF;
  }

  return decoded;
}
