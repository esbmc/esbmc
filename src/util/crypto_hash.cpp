
#include <boost/algorithm/hex.hpp>
#include <boost/uuid/sha1.hpp>
#include <cstring>
#include <util/crypto_hash.h>

class crypto_hash_private
{
public:
  boost::uuids::detail::sha1 s;
};

bool crypto_hash::operator<(const crypto_hash h2) const
{
  if(memcmp(hash, h2.hash, CRYPTO_HASH_SIZE) < 0)
    return true;

  return false;
}

std::string crypto_hash::to_string() const
{
  std::ostringstream buf;
  for(unsigned int i : hash)
    buf << std::hex << std::setfill('0') << std::setw(8) << i;

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
