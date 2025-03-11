export const convertToD3Tree = ({node, reference, index}) => {

  const currentKey = reference[index]
  const nextKey = reference[index + 1]
  
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

const snake = ["FIRST_NAME", "LAST_NAME", "LOGIN", "EMAIL",  "PASSWORD", "REP_PASSWORD", "REQUEST_DATE", "FILE_NAME", "STATUS", "RESULT",]
const camel = ["firstName", "lastName", "login", "email", "password", "repPassword", "request_date", "file_name", "status", "result", ]
const ru = ["Имя", "Фамилия", "Логин", "Электронная почта", "Пароль", "Повторите пароль", "Дата запроса", "Файл", "Статус", "Результат классификации",]

const combineKeys = (key1, key2) => {
  var result = {}
  
  key1.forEach((element, index) => {
    result[element] = key2[index]
  })

  return result
}

export const mappingInfo = combineKeys(snake, camel)
export const mappingInfoRU = combineKeys(camel, ru)

