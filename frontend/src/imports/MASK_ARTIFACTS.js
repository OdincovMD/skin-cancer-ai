import { CLASSIFICATION_ARTIFACT_FILE } from "./ENDPOINTS"
import { env } from "./ENV"

export const PROCESSING_MODE_CLASSIFICATION = "classification"
export const PROCESSING_MODE_MASK = "mask"

export const MASK_ARTIFACT_LABELS = {
  mask: {
    title: "Ч/б маска",
    filename: "mask.png",
    description: "Бинарная область новообразования",
  },
  masked_image: {
    title: "Маскированное изображение",
    filename: "masked_image.png",
    description: "Исходный снимок с затемнением вне маски",
  },
  archive: {
    title: "Архив",
    filename: "mask_results.zip",
    description: "Маска, маскированное изображение и manifest.json",
  },
}

export function isMaskResult(result) {
  return result != null && result.mode === PROCESSING_MODE_MASK
}

export function getMaskArtifacts(result) {
  if (!isMaskResult(result) || result.artifacts == null) {
    return {}
  }
  const artifacts = result.artifacts
  return {
    mask:
      artifacts.mask && typeof artifacts.mask === "object"
        ? artifacts.mask
        : null,
    masked_image:
      artifacts.masked_image && typeof artifacts.masked_image === "object"
        ? artifacts.masked_image
        : null,
    archive:
      artifacts.archive && typeof artifacts.archive === "object"
        ? artifacts.archive
        : null,
  }
}

export function artifactUrl(token) {
  if (!token) return null
  const base = env.BACKEND_URL.replace(/\/$/, "")
  return `${base}${CLASSIFICATION_ARTIFACT_FILE}?token=${encodeURIComponent(token)}`
}

export function hasMaskArtifacts(result) {
  const artifacts = getMaskArtifacts(result)
  return Boolean(artifacts.mask || artifacts.masked_image || artifacts.archive)
}
