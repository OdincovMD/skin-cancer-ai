import React from "react"
import Tree from "react-d3-tree"
import { classificationTree, finalResult } from "../imports/TREE"

var startingPoint = "Начало"

function getValues(obj) {
  if (typeof obj !== 'object' || obj === null) return [obj]; // Если не объект, вернуть как элемент списка

  return Object.values(obj).flatMap(getValues); // Рекурсивно собрать все значения
}

const convertToD3Tree = ({node, reference, index}) => {

  const currentKey = reference[index]
  const nextKey = reference[index + 1] // Добавить здесь код для обработки крайнего случая

  return {
    name: currentKey,
    children: Object.keys(node).map((key) => {
      if (key == nextKey) {
        return convertToD3Tree({node: node[nextKey], reference: reference, index: index + 1})
      } 
      else {
        return { name: key, children: [] }
      }
    })

  }
}

const TreeComponent = (classificationResult) => {

  const values = [startingPoint, ...getValues(classificationResult)]
  console.log(values)
  const treeData = convertToD3Tree({node: classificationTree, reference: values, index: 0})
  

  // console.log(classificationResult)
  // console.log(values)
  console.log(treeData)

  return (
    <div style={{ width: "100%", height: "500px" }}>
      <Tree data={treeData} orientation="horizontal" />
    </div>
  )
}

export default TreeComponent