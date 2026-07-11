#pragma once

#include <vector>

#include "pyplus/segmentation.hh" // struct Tip

namespace pyplus {

void split_merged_tips(std::vector<Tip> &tips, double merge_area_ratio = 1.6);

} // namespace pyplus
