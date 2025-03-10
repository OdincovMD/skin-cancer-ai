import React from "react"
import Tree from "react-d3-tree"
import { classificationTree } from "../imports/TREE"
import { getValues, convertToD3Tree } from "../imports/HELPERS"

const renderNode = ({ nodeDatum, toggleNode }) => {

  const toHighlight = (nodeDatum.children && (nodeDatum.children.length > 0)) || nodeDatum.attributes?.final
  const nodeColor = toHighlight ? "#99ff99" : "white"
  const fontWeight = toHighlight ? 600 : 400
  const color = toHighlight ? "black" : "gray"
  const strokeWidth = toHighlight ? "2" : "1"

  const textOffsetX = nodeDatum.attributes?.final ? 0 : 0
  const textOffsetY = nodeDatum.attributes?.final ? 60 : 0
  

  return (
    <g>
    <circle
      r="15"
      fill={nodeColor}
      stroke="#374151"
      strokeWidth={strokeWidth}
      onClick={toggleNode}
    />
    
    <text
      x={textOffsetX}
      y={-20 + textOffsetY}
      textAnchor="middle"
      fill="#374151"
      fontSize="24"
      strokeWidth={strokeWidth}
      fontWeight={fontWeight}
      color={color}
    >
      {nodeDatum.name}
    </text>
    </g>
  )
}

const TreeComponent = ({classificationResult, displaySize, nodeSize, zoom, translate}) => {

  const startingPoint = "Начало"
  const values = [startingPoint, ...getValues(classificationResult)]
  const treeData = convertToD3Tree({node: classificationTree, reference: values, index: 0})

  return (

    <div style={displaySize}>
      <Tree 
        data={treeData} 
        zoom={zoom}
        translate={translate}
        orientation="horizontal"
        nodeSize={nodeSize}
        separation={{ siblings: 2, nonSiblings: 3 }}
        renderCustomNodeElement={renderNode}
      />
    </div>
  )
}

export default TreeComponent