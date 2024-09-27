import base64

import numpy as np
import cv2
from roboflow import Roboflow
import os

import one
import ManyPatterns_DominantPattern

import one_structureless
import one_globules
import one_lines
import one_dots

import several_globules_symmetricOrAsymmetric
import two_circles
import several_lines
import several_dots

import final

from fastapi import FastAPI, File, UploadFile

app = FastAPI()


def main(path_to_img: str) -> list:
    image = cv2.imread(path_to_img)

    pred_one_or_more = "OnlyOne"  # TODO add model to classify one or more

    mask = "mask"  # TODO add model to segment neoplasm

    accumulate = []

    if pred_one_or_more == "OnlyOne":
        accumulate.append('Один признак')
        pred_one_which = one.main(image)

        if pred_one_which == 'Бесструктурная область':
            accumulate.append('Бесструктурная область')
            pred_one_structureless = one_structureless.main(image)

            if pred_one_structureless == 'Коричневый':
                accumulate.append('Коричневый')


            elif pred_one_structureless == 'Красный':
                accumulate.append('Красный')
            elif pred_one_structureless == 'Синий':
                accumulate.append('Синий')
            elif pred_one_structureless == 'Черный':
                accumulate.append('Черный')

        elif pred_one_which == 'Комки':
            accumulate.append("Комки")
            pred_one_globules = one_globules.main(image)
            if pred_one_globules == 'Желтый-белый':
                accumulate.append('Желтый-белый')
            elif pred_one_globules == 'Коричневый':
                accumulate.append('Коричневый')
            elif pred_one_globules == 'Красный':
                accumulate.append('Красный')
            elif pred_one_globules == 'Оранжевый':
                accumulate.append('Оранжевый')
            elif pred_one_globules == "Телесный":
                accumulate.append("Телесный")
            elif pred_one_globules == "Черный":
                accumulate.append("Черный")

        elif pred_one_which == "Круги":
            accumulate.append("Круги")
            accumulate.append('Продолжение ветки в разработке')


        elif pred_one_which == "Линии":
            accumulate.append("Линии")
            pred_one_lines = one_lines.main(image)
            if pred_one_lines == 'Curved':
                accumulate.append('Изогнутые')
            elif pred_one_lines == 'Parallel':
                accumulate.append('Параллельные')
            elif pred_one_lines == 'Reticular':
                accumulate.append('Ретикулярные')
            elif pred_one_lines == 'Spread':
                accumulate.append('Разветвленные')


        elif pred_one_which == "Точки":
            accumulate.append("Точки")
            pred_one_dots = one_dots.main(image, mask)
            if pred_one_dots == 'brown':
                accumulate.append('Коричневый')
            elif pred_one_dots == 'gray':
                accumulate.append('Серый')

    else:
        accumulate.append('Несколько признаков')
        tmp = ManyPatterns_DominantPattern.main(image)

        if tmp == 'Комки':
            accumulate.append("Комки")
            pred_several_globules = several_globules_symmetricOrAsymmetric.main(image, mask)
            if pred_several_globules == 'Симметричные':
                accumulate.append('СИММЕТРИЧНЫЕ')
            elif pred_several_globules == 'АСИММЕТРИЧНЫЕ':
                accumulate.append('Асимметричные')

        elif tmp == "Круги":
            accumulate.append("Круги")
            pred_several_circles = two_circles.main(image, mask)
            if pred_several_circles == 'Brown':
                accumulate.append('Коричневый')
            elif pred_several_circles == 'Black or Gray':
                accumulate.append('Черный или серый')

        elif tmp == "Линии":
            accumulate.append("Линии")
            pred_several_lines = number_of_signs_lines_full.main(image)
            if pred_several_lines == 'Curved':
                accumulate.append('Изогнутые')
            elif pred_several_lines == 'Parallel':
                accumulate.append('Параллельные')
            elif pred_several_lines == 'Radial':
                accumulate.append('Радиальные')
            elif pred_several_lines == 'Reticular_or_network':
                accumulate.append('Ретикулярные или разветвленные')

        elif tmp == "Точки":
            accumulate.append("Точки")
            pred_several_dots = several_dots.main(image)
            if pred_several_dots == 'Black':
                accumulate.append('Черный')
            elif pred_several_dots == 'Brown':
                accumulate.append('Коричневый')

    final_class = final.main(image)
    if final_class == 'Melanoma':
        accumulate.append('Меланома')
    elif final_class == 'Nevus':
        accumulate.append('Невус')
    elif final_class == 'BCC':
        accumulate.append('Базалиома')
    elif final_class == 'DF':
        accumulate.append('Дерматофиброма')
    elif final_class == 'SebK':
        accumulate.append('Себорейный кератоз')
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
