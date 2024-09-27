import cv2
import os

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

import one_structureless
import one_structureless_one_color
import one_structureless_more_than_one_color

import one_dots


import several

import several_lines
import several_lines_parallel
import several_lines_reticular
import several_lines_reticular_asymmetric
import several_lines_parallel_furrow
import several_lines_radial_pereferic  # не реализуется узел выше: несколько признаков -> линни -> радиальные 

import several_circles

import several_dots

import several_globules
import several_globules_asymmetric

import final

# from fastapi import FastAPI, File, UploadFile

# app = FastAPI()


def main(path_to_img: str) -> list:
    image = cv2.imread(path_to_img)

    pred_one_or_more = "Один"  # TODO add model to classify one or more

    mask = mask_builder.main(path_to_img) 

    accumulate = []

    def handle_globules(image):
        pred = one_globules.main(image)
        globules_color = {
            'single_color': 'Один цвет',
            'several_colors': 'Больше одного цвета'
        }
        accumulate.append(globules_color[pred])

        def handle_globules_one_color(image):
            pred = one_globules_one_color.main(image)
            colors = {
                'Желтый-белый': 'Желтый-белый',
                'Коричневый': 'Коричневый',
                'Красный': 'Красный',
                'Оранжевый': 'Оранжевый',
                'Телесный': 'Телесный',
                'Черный': 'Черный'
            }
            accumulate.append(colors[pred])

        def handle_globules_many_colors(image):
            pred = one_globules_many_color.main(image)
            colors = {
                'melanin': 'Меланин',
                'other': 'Другой пигмент'
            }
            accumulate.append(colors[pred])

            def handle_globules_many_colors_melanin(image):
                pred = one_globules_more_than_one_color_melanin.main(image)
                type_globules = {
                    'АСИММЕТРИЧНОЕ РАСПОЛОЖЕНИЕ ГЛОБУЛ': 'Ассиметричность',
                    'СИММЕТРИЧНОЕ РАСПОЛОЖЕНИЕ ГЛОБУЛ': 'Симметричность'
                }
                accumulate.append(type_globules[pred])

            one_globules_many_color_handlers = {
                'melanin': handle_globules_many_colors_melanin,
            }

            if pred in one_globules_many_color_handlers:
                one_globules_many_color_handlers[pred](image)

        globules_color = {
            'single_color': handle_globules_one_color,
            'several_colors': handle_globules_many_colors
        }
        globules_color[pred](image)

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
        pred = one_dots.main(image, mask)
        dot_colors = {
            'Коричневый': 'Коричневый',
            'Серый': 'Серый'
        }
        accumulate.append(dot_colors[pred])

    def handle_structureless(image):
        pred = one_structureless.main(image)
        one_structureless_type = {
            'monochrome': 'Один цвет',
            'multicolor': 'Несколько цветов'
        }
        accumulate.append(one_structureless_type[pred])

        def handle_one_structureless_one_color(image):
            pred = one_structureless_one_color.main(image)
            colors = {
            'Коричневый': 'Коричневый',
            'Красный': 'Красный',
            'Синий': 'Синий',
            'Черный': 'Черный'
            }
            accumulate.append(colors[pred])

        def handle_one_structureless_many_color(iamge):
            pred = one_structureless_more_than_one_color.main(image, mask)
            color = {
                'brown': 'Коричневый',
                'red': 'Красный',
                'yellow': 'Зелёный'
            }
            accumulate.append(color[pred])

        one_structureless_color_type = {
            'monochrome': handle_one_structureless_one_color,
            'multicolor': handle_one_structureless_many_color
        }

        one_structureless_color_type[pred](image)

    # Определение одного признака
    if pred_one_or_more == "Один":
        accumulate.append('Один признак')
        pred_one_which = one.main(image)

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
        else:
            accumulate.append("Псевдоподии")
            accumulate.append('Продолжение ветки в разработке')


    # Обработка нескольких признаков
    else:   
        accumulate.append('Несколько признаков')
        pred = several.main(image)

        def handle_several_globules(image):
            pred = several_globules.main(image, mask)
            symmetry_types = {
                'СИММЕТРИЧНЫЕ': 'Симметричные',
                'АСИММЕТРИЧНЫЕ': 'Асимметричные'
            }
            accumulate.append(symmetry_types[pred])

            def handle_several_globules_asymmetric(image):
                pred = several_globules_asymmetric.main(image)
                color = {
                    'Другой': 'Другой',
                    'Меланин': 'Меланин'
                }
                accumulate.append(color[pred])

            several_globules_handlers ={
                'АСИММЕТРИЧНЫЕ': handle_several_globules_asymmetric
            }

            if pred in several_globules_handlers:
                several_globules_handlers[pred](image)

        def handle_several_lines(image):
            pred = several_lines.main(image)
            line_types = {
                'Curved': 'Изогнутые',
                'Parallel': 'Параллельные',
                'Radial': 'Радиальные',
                'Reticular_or_network': 'Ретикулярные или разветвленные'
            }
            accumulate.append(line_types[pred])

            def handle_several_lines_parallel(image):
                pred = several_lines_parallel.main(image)
                type_parallel_lines = {
                    'Borozd': 'Борозды',
                    'Grebesh': 'Гребешки'
                }
                accumulate.append(type_parallel_lines[pred])

                def handle_several_lines_parallel_furrow(image):
                    pred = several_lines_parallel_furrow.main(image)
                    type_parallel_lines_furrow = {
                        'Симметрия': 'Симметрия',
                        'Асимметрия': 'Асимметрия'
                    }
                    accumulate.append(type_parallel_lines_furrow[pred])

                type_parallel_lines_handlers = {
                    'Borozd': handle_several_lines_parallel_furrow
                }

                if pred in type_parallel_lines_handlers:
                    type_parallel_lines_handlers[pred](image)
                
            def handle_several_lines_reticular(image):
                pred =  several_lines_reticular.main(image)
                type_reticular_lines = {
                    'Asymmetric': 'Ассиметричные',
                    'Symmetric': 'Симметричные'
                }
                accumulate.append(type_reticular_lines[pred])

                def handle_several_lines_reticular_assymetric(image):
                    pred = several_lines_reticular_asymmetric.main(image)
                    colors = {
                        'ОДИН ЦВЕТ': 'Один цвет',
                        'БОЛЬШЕ ОДНОГО ЦВЕТА': 'Больше одного цвета'
                    }

                    accumulate.append(colors[pred])

                several_lines_reticular_handlers = {
                    'Asymmetric': handle_several_lines_reticular_assymetric
                }

                if pred in several_lines_reticular_handlers:
                    several_lines_reticular_handlers[pred](image)


            several_lines_handler = {
                'Parallel': handle_several_lines_parallel,
                'Reticular_or_network': handle_several_lines_reticular
            }
            if pred in several_lines_handler:
                several_lines_handler[pred](image)

        def handle_several_circles(image):
            pred = several_circles.main(image, mask)
            circles_type = {
                'Brown': 'Коричневый',
                'Black or Gray': 'Черный или серый'
            }
            accumulate.append(circles_type[pred])
        
        def handle_several_dots(image):
            pred = several_dots.main(image)
            several_dots_type = {
                'Black': 'Черный',
                'Brown': 'Коричневый'
            }
            accumulate.append(several_dots_type[pred])

        several_handlers = {
            'Комки': handle_several_globules,
            'Круги': handle_several_circles,
            'Линии': handle_several_lines,
            'Точки': handle_several_dots
        }

        
        accumulate.append(pred)
        several_handlers[pred](image)

    # Финальная классификация
    pred = final.main(image)
    final_classes = {
        'Melanoma': 'Меланома',
        'Nevus': 'Невус',
        'BCC': 'Базалиома',
        'DF': 'Дерматофиброма',
        'SebK': 'Себорейный кератоз'
    }
    accumulate.append(final_classes[pred])

    return accumulate

# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile = File(...)):
#     with open(file.filename, "wb") as buffer:
#         buffer.write(file.file.read())
#     result = main(file.filename)
#     os.remove(file.filename)
#     return result

if __name__ == '__main__':
    main('26.jpg')
