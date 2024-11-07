import cv2
import one_several
import mask_builder

import one
import one_lines
import one_lines_reticular
import one_lines_branched
import one_lines_parallel
import one_lines_reticular_one_color_black_or_brown
import one_lines_reticular_more_than_one_color

import one_globules
import one_globules_one_color
import one_globules_many_color
import one_globules_more_than_one_color_melanin

# import one_structureless
import one_structureless_one_color
import one_structureless_more_than_one_color

import one_dots

import several
import several_lines
import several_lines_parallel
import several_lines_reticular
import several_lines_reticular_asymmetric
import several_lines_parallel_furrow
import several_lines_radial_pereferic
import several_circles
import several_dots
import several_globules
import several_globules_asymmetric
import final

debug_list = list()
path_to_img = '26.jpg'

# Загрузка изображения
try:
    image = cv2.imread(path_to_img)
except Exception as e:
    debug_list.append(("Image Loading", f"Error: {e}"))

# Создание маски
try:
    mask = mask_builder.main(path_to_img)
    debug_list.append(("Mask Generation", mask))
except Exception as e:
    debug_list.append(("Mask Generation", f"Error: {e}"))

# Выполнение модулей и функций
modules = [
    ("One or Several Features", one_several.main, [image]),
    ("One - Main", one.main, [image]),
    ("One - Lines", one_lines.main, [image]),
    ("One - Reticular Lines", one_lines_reticular.main, [image]),
    ("One - Branched Lines", one_lines_branched.main, [image], {"mask": mask}),
    ("One - Parallel Lines", one_lines_parallel.main, [image]),
    ("One - Reticular One Color (Black/Brown)", one_lines_reticular_one_color_black_or_brown.main, [image]),
    ("One - Reticular More Than One Color", one_lines_reticular_more_than_one_color.main, [image]),
    ("One - Globules", one_globules.main, [image]),
    ("One - Globules One Color", one_globules_one_color.main, [image]),
    ("One - Globules Many Color", one_globules_many_color.main, [image]),
    ("One - Globules More Than One Color (Melanin)", one_globules_more_than_one_color_melanin.main, [image]),
    # ("One - Structureless", one_structureless.main, [image]),
    ("One - Structureless One Color", one_structureless_one_color.main, [image]),
    ("One - Structureless More Than One Color", one_structureless_more_than_one_color.main, [image], {"mask": mask}),
    ("One - Dots", one_dots.main, [image], {"mask": mask}),
    ("Several - Main", several.main, [image]),
    ("Several - Lines", several_lines.main, [image], {"mask": mask}),
    ("Several - Parallel Lines", several_lines_parallel.main, [image]),
    ("Several - Reticular Lines", several_lines_reticular.main, [image]),
    ("Several - Reticular Asymmetric Lines", several_lines_reticular_asymmetric.main, [image], {"mask": mask}),
    ("Several - Parallel Furrow Lines", several_lines_parallel_furrow.main, [image], {"mask": mask}),
    ("Several - Radial Peripheral Lines", several_lines_radial_pereferic.main, [image], {"mask": mask}),
    ("Several - Circles", several_circles.main, [image], {"mask": mask}),
    ("Several - Dots", several_dots.main, [image]),
    ("Several - Globules", several_globules.main, [image], {"mask": mask}),
    ("Several - Asymmetric Globules", several_globules_asymmetric.main, [image]),
]

for name, func, args, kwargs in [(m[0], m[1], m[2], m[3] if len(m) > 3 else {}) for m in modules]:
    try:
        result = func(*args, **kwargs)
        debug_list.append((name, result))
    except Exception as e:
        debug_list.append((name, f"Error: {e}"))

try:
    final_result = final.main(image)
    debug_list.append(("Final Result", final_result))
except Exception as e:
    debug_list.append(("Final Result", f"Error: {e}"))

print(debug_list)
