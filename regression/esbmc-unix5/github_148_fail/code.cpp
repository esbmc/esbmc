    else if(is_array_type(it))
    {
      std::cout << "### it: " << std::endl;
      it->dump();
      std::cerr << "Fetching array elements inside tuples currently "
                   "unimplemented, sorry"
                << std::endl;
      tuple->elements[i]->dump();
      //res = ctx->get_array(tuple->elements[i]);

      // Check size
      const array_type2t &arr_type = to_array_type(it);
      if(arr_type.size_is_infinite)
      {
        // Guarentee nothing, this is modelling only.
        std::cout << "passou aqui" << std::endl;
        assert(0);
      }
      if(!is_constant_int2t(arr_type.array_size))
      {
        std::cerr << "Non-constant sized array of type constant_array_of2t"
                  << std::endl;
        assert(0);
      }
      std::cout << "### arr_type: " << std::endl;
      arr_type.dump();
      res =
        ctx->array_api->get_array_elem(tuple->elements[i], 0, arr_type.subtype);
      std::cout << "### res: " << std::endl;
      res->dump();
    }
