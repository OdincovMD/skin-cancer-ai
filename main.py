import cv2

import one_several
import mask_builder

import one

import one_lines
import one_lines_reticular
import one_lines_branched
import one_lines_parallel
import one_lines_reticular_one_color
import one_lines_reticular_several_colors

import one_globules
import one_globules_one_color
import one_globules_several_colors
import one_globules_several_colors_melanin

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
import several_globules_asymmetrical
import several_globules_asymmetrical_melanin

import final

# from fastapi import FastAPI, File, UploadFile

# app = FastAPI()


def main(path_to_img: str) -> list:
    image = cv2.imread(path_to_img)

    pred = one_several.main(image)

    mask = mask_builder.main(path_to_img) 

    accumulate = []

    def handle_globules(image):
        pred = one_globules.main(image)
        globules_color = {
            'Один цвет': 'Один цвет',
            'Более одного цвета': 'Более одного цвета'
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
            pred = one_globules_several_colors.main(image)
            colors = {
                'Меланин': 'Меланин',
                'Другой пигмент': 'Другой пигмент'
            }
            accumulate.append(colors[pred])

            def handle_globules_many_colors_melanin(image):
                pred = one_globules_several_colors_melanin.main(image)
                type_globules = {
                    'Цвета расположены асимметрично': 'Цвета расположены асимметрично',
                    'Цвета расположены симметрично': 'Цвета расположены симметрично'
                }
                accumulate.append(type_globules[pred])

            one_globules_many_color_handlers = {
                'Меланин': handle_globules_many_colors_melanin,
            }

            if pred in one_globules_many_color_handlers:
                one_globules_many_color_handlers[pred](image)

        globules_color = {
            'Один цвет': handle_globules_one_color,
            'Более одного цвета': handle_globules_many_colors
        }
        globules_color[pred](image)

    def handle_one_lines(image):
        pred = one_lines.main(image)
        line_types = {
            'Изогнутые': 'Изогнутые',
            'Параллельные': 'Параллельные',
            'Ретикулярные': 'Ретикулярные',
            'Разветвленные': 'Разветвленные'
        }
        accumulate.append(line_types[pred])

        def handle_one_lines_reticular(image):
            pred = one_lines_reticular.main(image)
            reticular_type = {
                'Один цвет': 'Один цвет',
                'Больше одного цвета': 'Больше одного цвета'
            }
            accumulate.append(reticular_type[pred])

            def handle_one_lines_reticular_one_color(image):
                pred = one_lines_reticular_one_color.main(image)
                color = {
                    'Черные': 'Черный',
                    'Коричневые': 'Коричневый'
                }
                accumulate.append(color[pred])

            def handle_one_lines_reticular_more_one_color(image):
                pred = one_lines_reticular_several_colors.main(image)
                color = {
                    'Пестрый и краптовый': 'Пестрый или краптовый',
                    'Центральная гиперпигментация': 'Центральная гиперпигментация',
                    'Периферическая гиперпигментация': 'Периферическая гиперпигментация', 
                }
                accumulate.append(color[pred])

            one_lines_reticular_color_handlers = {
                'Один цвет': handle_one_lines_reticular_one_color,
                'Больше одного цвета': handle_one_lines_reticular_more_one_color
            }
            accumulate.append(one_lines_reticular_color_handlers[pred])

        
        def handle_one_lines_branched(image):
            pred = one_lines_branched.main(image, mask=mask)
            branched_type = {
                'Коричневые': 'Коричневый',
                'Черные': 'Чёрный'
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
            'Ретикулярные': handle_one_lines_reticular,
            'Разветвленные': handle_one_lines_branched,
            'Параллельные': handle_one_lines_parallel,
        }
        if pred in one_lines_handlers:
            one_lines_handlers[pred](image)
    
    def handle_dots(image):
        pred = one_dots.main(image, mask=mask)
        dot_colors = {
            'Коричневый': 'Коричневый',
            'Серый': 'Серый'
        }
        accumulate.append(dot_colors[pred])

    def handle_structureless(image):
        pred = one_structureless.main(image)
        one_structureless_type = {
            'Один цвет': 'Один цвет',
            'Несколько цветов': 'Несколько цветов'
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

        def handle_one_structureless_many_color(image):
            pred = one_structureless_more_than_one_color.main(image, mask=mask)
            color = {
                'Коричневый': 'Коричневый',
                'Красный': 'Красный',
                'Желтый': 'Желтый'
            }
            accumulate.append(color[pred])

        one_structureless_color_type = {
            'Один цвет': handle_one_structureless_one_color,
            'Несколько цветов': handle_one_structureless_many_color
        }

        one_structureless_color_type[pred](image)

    def handle_several_globules(image):
        pred = several_globules.main(image, mask=mask)
        symmetry_types = {
            'Симметричные': 'Симметричные',
            'Асимметричные': 'Асимметричные'
        }
        accumulate.append(symmetry_types[pred])

        def handle_several_globules_asymmetric(image):
            pred = several_globules_asymmetrical.main(image)
            color = {
                'Другой': 'Другой',
                'Меланин': 'Меланин'
            }
            accumulate.append(color[pred])

            def handle_several_globules_asymmetric_melanin(image):
                pred = several_globules_asymmetrical_melanin(image, mask)
                color = {
                    'Больше одного цвета': 'Больше одного цвета', 
                    'Один цвет (коричневый)': 'Один цвет (коричневый)'
                }
                accumulate.append(color[pred])
            several_globules_asymmetrical_handled = {
                'Меланин': handle_several_globules_asymmetric_melanin,
                'Другой': lambda: 'Другой'
            }
            if pred in several_globules_asymmetrical_handled:
                several_globules_asymmetrical_handled[pred](image)

        several_globules_handlers ={
            'Асимметричные': handle_several_globules_asymmetric
        }

        if pred in several_globules_handlers:
            several_globules_handlers[pred](image)

    def handle_several_lines(image):
        pred = several_lines.main(image, mask=mask)
        line_types = {
            'Изогнутые': 'Изогнутые',
            'Параллельные': 'Параллельные',
            'Радиальные': 'Радиальные',
            'Ретикулярные или разветвленные': 'Ретикулярные или разветвленные'
        }
        accumulate.append(line_types[pred])

        def handle_several_lines_parallel(image):
            pred = several_lines_parallel.main(image)
            type_parallel_lines = {
                'Борозды': 'Борозды',
                'Гребешки': 'Гребешки'
            }
            accumulate.append(type_parallel_lines[pred])

            def handle_several_lines_parallel_furrow(image):
                pred = several_lines_parallel_furrow.main(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask=mask)
                type_parallel_lines_furrow = {
                    'Симметрия': 'Симметрия',
                    'Асимметрия': 'Асимметрия'
                }
                accumulate.append(type_parallel_lines_furrow[pred])

            type_parallel_lines_handlers = {
                'Борозды': handle_several_lines_parallel_furrow
            }

            if pred in type_parallel_lines_handlers:
                type_parallel_lines_handlers[pred](image)
            
        def handle_several_lines_reticular(image):
            pred =  several_lines_reticular.main(image)
            type_reticular_lines = {
                'Ассиметричные': 'Ассиметричные',
                'Симметричные': 'Симметричные'
            }
            accumulate.append(type_reticular_lines[pred])

            def handle_several_lines_reticular_assymetric(image):
                pred = several_lines_reticular_asymmetric.main(image, mask=mask)
                colors = {
                    'Один цвет': 'Один цвет',
                    'Больше одного цвета': 'Больше одного цвета'
                }

                accumulate.append(colors[pred])

            several_lines_reticular_handlers = {
                'Ассиметричные': handle_several_lines_reticular_assymetric
            }

            if pred in several_lines_reticular_handlers:
                several_lines_reticular_handlers[pred](image)


        several_lines_handler = {
            'Параллельные': handle_several_lines_parallel,
            'Ретикулярные или разветвленные': handle_several_lines_reticular
        }
        if pred in several_lines_handler:
            several_lines_handler[pred](image)

    def handle_several_circles(image):
        pred = several_circles.main(image, mask=mask)
        circles_type = {
            'Коричневый': 'Коричневый',
            'Черный или серый': 'Черный или серый'
        }
        accumulate.append(circles_type[pred])
    
    def handle_several_dots(image):
        pred = several_dots.main(image, mask=mask)
        several_dots_type = {
            'Черный': 'Черный',
            'Коричневый': 'Коричневый'
        }
        accumulate.append(several_dots_type[pred])

    # Определение одного признака
    if pred == "Один":
        accumulate.append('Один признак')
        pred = one.main(image)

        structure_handlers = {
            'Бесструктурная область': handle_structureless,
            'Комки': handle_globules,
            'Линии': handle_one_lines,
            'Точки': handle_dots,
        }

        if pred in structure_handlers:
            accumulate.append(pred)
            structure_handlers[pred](image)
        else:
            accumulate.append("Круги либо Псевдоподии")
            accumulate.append('Продолжение ветки в разработке')

    # Обработка нескольких признаков
    else:   
        accumulate.append('Несколько признаков')
        pred = several.main(image)
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
