
#include <boost/algorithm/hex.hpp>
#include <boost/version.hpp>
#if BOOST_VERSION >= 106600
#  include <boost/uuid/detail/sha1.hpp>
#else
#  include <boost/uuid/sha1.hpp>
#endif
#include <cstring>
#include <util/crypto_hash.h>

class crypto_hash_private
{
public:
  boost::uuids::detail::sha1 s;
};

bool crypto_hash::operator<(const crypto_hash &h2) const
{
  if (memcmp(hash, h2.hash, sizeof(hash)) < 0)
    return true;

  return false;
}

std::string crypto_hash::to_string() const
{
  std::ostringstream buf;
#if BOOST_VERSION >= 108600
  for (HashType c : hash)
  {
    buf << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(c);
  }
#else
  for (HashType i : hash)
    buf << std::hex << std::setfill('0') << std::setw(8) << i;
#endif

  return buf.str();
}

crypto_hash::crypto_hash()
  : p_crypto(std::make_shared<crypto_hash_private>()), hash{0}
{
}

void crypto_hash::ingest(void const *data, unsigned int size)
{
  p_crypto->s.process_bytes(data, size);
}

void crypto_hash::fin()
{
  p_crypto->s.get_digest(hash);
}
