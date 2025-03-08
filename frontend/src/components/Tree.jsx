import React from "react"
import Tree from "react-d3-tree"

const treeData = {
    name: "Корень",
    children: [
      {
        name: "Ветка 1",
        children: [{ name: "Лист 1" }, { name: "Лист 2" }],
      },
      {
        name: "Ветка 2",
        children: [{ name: "Лист 3" }],
      },
    ],
  }

const TreeComponent = () => {
  return (
    <div style={{ width: "100%", height: "500px" }}>
      <Tree data={treeData} orientation="horizontal" />
    </div>
  )
}

export default TreeComponent