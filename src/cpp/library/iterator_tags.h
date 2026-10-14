// The [iterator.tags] category tags, shared by <iterator> and <string>.
//
// Lifted out of <iterator> unchanged. They lived there, but <iterator> includes
// <iostream>, which reaches <string>, so <string> cannot include <iterator>
// back -- and without the tags a string iterator cannot name its
// iterator_category, which [iterator.traits] requires. Same reason char_traits
// was extracted in #7050.
#pragma once

namespace std
{
/* [iterator.tags]: the five categories, with the inheritance the standard
 * specifies. forward_ and bidirectional_ were missing, and random_access_ did
 * not derive from anything, so tag dispatch could not select an overload. */
///  Marking input iterators.
struct input_iterator_tag
{
};

///  Marking output iterators.
struct output_iterator_tag
{
};

///  Marking forward iterators.
struct forward_iterator_tag : public input_iterator_tag
{
};

///  Marking bidirectional iterators.
struct bidirectional_iterator_tag : public forward_iterator_tag
{
};

// Marking random iterators
struct random_access_iterator_tag : public bidirectional_iterator_tag
{
};
} // namespace std
