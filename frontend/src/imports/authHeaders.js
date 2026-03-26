/** @param {string|null|undefined} accessToken */
export function bearerAuthHeaders(accessToken) {
  if (!accessToken) return {}
  return { Authorization: `Bearer ${accessToken}` }
}
