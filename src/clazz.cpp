#include "clazz.hpp"

void clazz::add(int k)
{
    std::transform(vec_.begin(), vec_.end(), vec_.begin(), [&](const auto& v) { return v + k; });
}
