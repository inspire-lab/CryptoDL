/*
 * DarkMagic.h
 *
 *  Created on: May 10, 2019
 *      Author: robert
 */

#ifndef ARCHITECTURE_DARKMAGIC_H_
#define ARCHITECTURE_DARKMAGIC_H_

#include <vector>
#include <type_traits>

//specialize a type for all of the STL containers.
namespace is_stl_container_impl{
  template <typename T>       struct is_vector:std::false_type{};
  template <typename... Args> struct is_vector<std::vector <Args...>>:std::true_type{};
}

//type trait to utilize the implementation type traits as well as decay the type
template <typename T> struct is_vector {
	static constexpr bool const value = is_stl_container_impl::is_vector<typename std::decay<T>::type>::value;

};



#endif /* ARCHITECTURE_DARKMAGIC_H_ */
