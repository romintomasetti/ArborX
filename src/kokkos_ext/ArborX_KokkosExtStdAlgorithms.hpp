/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_KOKKOS_EXT_STD_ALGORITHMS_HPP
#define ARBORX_KOKKOS_EXT_STD_ALGORITHMS_HPP

#include <kokkos_ext/ArborX_KokkosExtAccessibilityTraits.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

namespace ArborX::Details::KokkosExt
{

using Kokkos::Experimental::adjacent_difference;
using Kokkos::Experimental::exclusive_scan;
using Kokkos::Experimental::reduce;

template <typename ViewType>
struct Iota
{
    ViewType data;
    typename ViewType::value_type value;
    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index) const {
        data(index) = value + (typename ViewType::value_type)index;
    }

    template <typename Exec>
    auto apply(Exec&& exec) const
    {
        return std::forward<Exec>(exec) | Kokkos::Experimental::graph::parallel_for(
            "ArborX::Algorithms::iota",
            Kokkos::RangePolicy(/* space,*/ 0, data.extent(0)),
            *this
        );
    }
};

template <typename ExecutionSpace, typename ViewType>
auto iota(ExecutionSpace const &space, ViewType const &v,
          typename ViewType::value_type value = 0)
{
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);
  static_assert(Kokkos::is_view<ViewType>::value);
  static_assert(is_accessible_from<typename ViewType::memory_space,
                                   ExecutionSpace>::value,
                "View must be accessible from the execution space");
  static_assert(unsigned(ViewType::rank()) == unsigned(1),
                "iota requires a View of rank 1");

  using ValueType = typename ViewType::value_type;
  static_assert(std::is_arithmetic_v<ValueType>,
                "iota requires a View with an arithmetic value type");
  static_assert(
      std::is_same_v<ValueType, typename ViewType::non_const_value_type>,
      "iota requires a View with non-const value type");

//   return space | Kokkos::parallel_for(
//       "ArborX::Algorithms::iota", Kokkos::RangePolicy(space, 0, v.extent(0)),
//       KOKKOS_LAMBDA(int i) { v(i) = value + (ValueType)i; });
    return Iota{.data = v, .value = value}.apply(space);
}

} // namespace ArborX::Details::KokkosExt

#endif
