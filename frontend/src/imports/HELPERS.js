export const convertToD3Tree = ({node, reference, index}) => {

  const currentKey = reference[index]
  const nextKey = reference[index + 1]
  
  console.log(node, nextKey)
  
  const currentLeaf = node[nextKey]
  ? {
    name: currentKey,
    children: Object.keys(node).map((key) => {
    return key === nextKey 
      ?  convertToD3Tree({node: node[nextKey], reference: reference, index: index + 1})
      :  { name: key, children: [] }
    })
  } : {
    name: currentKey,
    children: [{
      name: nextKey,
      attributes: {
        final: true
      },
      children: []
    }]
  }
  
  return currentLeaf
}

export const getValues = (obj) => {
  if (typeof obj !== 'object' || obj === null) return [obj]
  
  return Object.values(obj).flatMap(getValues)
}

