
#include "crypto_hash.h"

#include <boost/algorithm/hex.hpp>
#include <boost/uuid/sha1.hpp>

#include <cstring>

class crypto_hash_private {
public:
  boost::uuids::detail::sha1 s;
};

bool crypto_hash::operator<(const crypto_hash h2) const
{

  if (memcmp(hash, h2.hash, CRYPTO_HASH_SIZE) < 0)
    return true;

  return false;
}

std::string crypto_hash::to_string() const
{
  return std::string(hash);
}

crypto_hash::crypto_hash()
  : p_crypto(std::make_shared<crypto_hash_private>()),
    hash{0}
{
}

void crypto_hash::ingest(void const *data, unsigned int size)
{
  p_crypto->s.process_bytes(data, size);
}

void crypto_hash::fin()
{
  unsigned int digest[5];
  p_crypto->s.get_digest(digest);

  for (int i = 0; i < 5; i++)
    std::sprintf(hash + (i << 3), "%08x", digest[i]);
}
