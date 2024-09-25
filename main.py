import cv2
import os

import mask_builder

import one
import several

import one_structureless
import one_globules
import one_dots

import one_lines
import one_lines_reticular
import one_lines_branched
import one_lines_parallel
import one_lines_reticular_one_color_black_or_brown
import one_lines_reticular_more_than_one_color

import several_globules_symmetricOrAsymmetric
import two_circles
import number_of_signs_lines_full
import several_dots

import final

from fastapi import FastAPI, File, UploadFile

app = FastAPI()


def main(path_to_img: str) -> list:
    image = cv2.imread(path_to_img)

    pred_one_or_more = "Один"  # TODO add model to classify one or more

    mask = mask_builder.main(path_to_img) 

    accumulate = []

    def handle_structureless(image):
        pred_one_structureless = one_structureless.main(image)
        colors = {
            'Коричневый': 'Коричневый',
            'Красный': 'Красный',
            'Синий': 'Синий',
            'Черный': 'Черный'
        }
        accumulate.append(colors[pred_one_structureless])

    def handle_globules(image):
        pred = one_globules.main(image)
        colors = {
            'Желтый-белый': 'Желтый-белый',
            'Коричневый': 'Коричневый',
            'Красный': 'Красный',
            'Оранжевый': 'Оранжевый',
            'Телесный': 'Телесный',
            'Черный': 'Черный'
        }
        accumulate.append(colors[pred])

    def handle_one_lines(image):
        pred = one_lines.main(image)
        line_types = {
            'Curved': 'Изогнутые',
            'Parallel': 'Параллельные',
            'Reticular': 'Ретикулярные',
            'Spread': 'Разветвленные'
        }
        accumulate.append(line_types[pred])

        def handle_one_lines_reticular(image):
            pred = one_lines_reticular.main(image)
            reticular_type = {
                '1_color': 'Один цвет',
                'more_colors': 'Больше одного цвета'
            }
            accumulate.append(reticular_type[pred])

            def handle_one_lines_reticular_one_color(image):
                pred = one_lines_reticular_one_color_black_or_brown.main(image)
                color = {
                    'ЧЕРНЫЕ': 'Черный',
                    'КОРИЧНЕВЫЕ': 'Коричневый'
                }
                accumulate.append(color[pred])

            def handle_one_lines_reticular_more_one_color(image):
                pred = one_lines_reticular_more_than_one_color.main(image)
                color = {
                    'ПЕСТРЫЙ ИЛИ КРАПОВЫЙ': 'Пестрый или краптовый',
                    'ЦЕНТРАЛЬНАЯ ГИПЕРПИГМЕНТАЦИЯ': 'Центральная гиперпигментация',
                    'ПЕРИФЕРИЧЕСКАЯ ГИПЕРПИГМЕНТАЦИЯ': 'Периферическая гиперпигментация', 
                }
                accumulate.append(color[pred])

            one_lines_reticular_color_handlers = {
                '1_color': handle_one_lines_reticular_one_color,
                'more_colors': handle_one_lines_reticular_more_one_color
            }
            accumulate.append(one_lines_reticular_color_handlers[pred])

        
        def handle_one_lines_branched(image):
            pred = one_lines_branched.main(image)
            branched_type = {
                'brown': 'Коричневый',
                'black': 'Чёрный'
            }
            accumulate.append(branched_type[pred])
        
        def handle_one_lines_parallel(image):
            pred = one_lines_parallel.main(image)
            parallel_type = {
                'Борозды': 'Борозды',
                'Гребешки': 'Гребешки',
                'Пересекающиеся гребешки и борозды': 'Пересекающиеся гребешки и борозды'
            }
            accumulate.append(parallel_type[pred])

        one_lines_handlers = {
            'Reticular': handle_one_lines_reticular,
            'Spread': handle_one_lines_branched,
            'Parallel': handle_one_lines_parallel,
        }
        if pred in one_lines_handlers:
            one_lines_handlers[pred](image)
    
    def handle_dots(image):
        pred_one_dots = one_dots.main(image, mask)
        dot_colors = {
            'brown': 'Коричневый',
            'gray': 'Серый'
        }
        accumulate.append(dot_colors[pred_one_dots])

    # Определение одного признака
    if pred_one_or_more == "Один":
        accumulate.append('Один признак')
        pred_one_which = one.ma in(image)

        structure_handlers = {
            'Бесструктурная область': handle_structureless,
            'Комки': handle_globules,
            'Линии': handle_one_lines,
            'Точки': handle_dots,
        }

        if pred_one_which in structure_handlers:
            accumulate.append(pred_one_which)
            structure_handlers[pred_one_which](image)
        elif pred_one_which == "Круги":
            accumulate.append("Круги")
            accumulate.append('Продолжение ветки в разработке')

    # Обработка нескольких признаков
    else:
        accumulate.append('Несколько признаков')
        main_feature = several.main(image)

        def handle_several_globules(image, mask):
            pred_several_globules = several_globules_symmetricOrAsymmetric.main(image, mask)
            symmetry_types = {
                'СИММЕТРИЧНЫЕ': 'Симметричные',
                'АСИММЕТРИЧНЫЕ': 'Асимметричные'
            }
            accumulate.append(symmetry_types[pred_several_globules])

        def handle_several_lines(image):
            pred_several_lines = number_of_signs_lines_full.main(image)
            line_types = {
                'Curved': 'Изогнутые',
                'Parallel': 'Параллельные',
                'Radial': 'Радиальные',
                'Reticular_or_network': 'Ретикулярные или разветвленные'
            }
            accumulate.append(line_types[pred_several_lines])

        def handle_two_circles(image):
            pred_several_circles = two_circles.main(image, mask)
            circles_type = {
                'Brown': 'Коричневый',
                'Black or Gray': 'Черный или серый'
            }
            accumulate.append(circles_type[pred_several_circles])
        
        def handle_several_dots(image):
            pred_several_dots = several_dots.main(image)
            several_dots_type = {
                'Black': 'Черный',
                'Brown': 'Коричневый'
            }
            accumulate.append(several_dots_type[pred_several_dots])


        several_handlers = {
            'Комки': handle_several_globules,
            'Круги': handle_two_circles,
            'Линии': handle_several_lines,
            'Точки': handle_several_dots
        }

        if main_feature in several_handlers:
            accumulate.append(main_feature)
            several_handlers[main_feature](image)

    # Финальная классификация
    final_class = final.main(image)
    final_classes = {
        'Melanoma': 'Меланома',
        'Nevus': 'Невус',
        'BCC': 'Базалиома',
        'DF': 'Дерматофиброма',
        'SebK': 'Себорейный кератоз'
    }
    accumulate.append(final_classes[final_class])

    return accumulate

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    with open(file.filename, "wb") as buffer:
        buffer.write(file.file.read())
    result = main(file.filename)
    os.remove(file.filename)
    return result

# if __name__ == '__main__':
#     main('26.jpg')
