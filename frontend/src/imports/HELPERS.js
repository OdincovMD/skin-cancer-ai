export const convertToD3Tree = ({
  node,
  reference,
  index,
  depth = 0,
  includeBranches = true,
}) => {

  const currentKey = reference[index]
  const nextKey = reference[index + 1]
  
  const currentLeaf = node[nextKey]
    ? {
        name: currentKey,
        attributes: {
          state: "active",
          depth,
        },
        children: Object.keys(node)
          .filter((key) => includeBranches || key === nextKey)
          .map((key) => {
            return key === nextKey
              ? convertToD3Tree({
                  node: node[nextKey],
                  reference: reference,
                  index: index + 1,
                  depth: depth + 1,
                  includeBranches,
                })
              : {
                  name: key,
                  attributes: {
                    state: "branch",
                    depth: depth + 1,
                  },
                  children: [],
                }
          }),
      }
    : {
        name: currentKey,
        attributes: {
          state: "active",
          depth,
        },
        children: [
          {
            name: nextKey,
            attributes: {
              final: true,
              state: "final",
              depth: depth + 1,
            },
            children: [],
          },
        ],
      }
  
  return currentLeaf
}

export const getValues = (obj) => {
  if (typeof obj !== 'object' || obj === null) return [obj]
  
  return Object.values(obj).flatMap(getValues)
}

const snake = ["EMAIL", "PASSWORD", "REP_PASSWORD", "REQUEST_DATE", "FILE_NAME", "STATUS", "RESULT"]
const camel = ["email", "password", "repPassword", "request_date", "file_name", "status", "result"]

const combineKeys = (key1, key2) => {
  var result = {}
  
  key1.forEach((element, index) => {
    result[element] = key2[index]
  })

  return result
}

export const mappingInfo = combineKeys(snake, camel)
