export const env = {
  BACKEND_URL: import.meta.env.VITE_BACKEND_URL || "/_api",
  VK_ID_APP_ID: import.meta.env.VITE_VK_ID_APP_ID || "",
  VK_ID_REDIRECT_URI: import.meta.env.VITE_VK_ID_REDIRECT_URI || "",
  VK_ID_SCOPES: import.meta.env.VITE_VK_ID_SCOPES || "vkid.personal_info email",
}
