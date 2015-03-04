// A test that causes the 'fgasdf' template to be instantiated, and then the
// constructor for that class (which is also a template) to be instantiated.
// This checks that the outer template argument (X) can be accessed from within
// the context where the constructor is instantiated.

//#include <iostream>

template <class X>
class fgasdf
{
  public:
    template <class Y>
    fgasdf(const Y &b)
    {
      X huuurrr = 'a';
      qux = b;
    }
    X qux;
};

int
main()
{
  int wuwuwuwu = 5168;
  fgasdf<char> tralala(wuwuwuwu);
//  std::cerr << tralala.qux << std::endl;
  return 0;
}
