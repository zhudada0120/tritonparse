/**
 * Utility functions for IR language display
 * Fully dynamic implementation using backend metadata
 */

import type { ProcessedKernel } from "./dataLoader";

/**
 * Get a user-friendly display name for the IR language
 * Uses metadata from ir_stages if available, otherwise generates reasonable defaults
 * @param irType - The type/name of IR file
 * @param kernel - Optional kernel metadata for display name lookup
 * @returns A human-readable language name
 */
export const getDisplayLanguage = (irType: string, kernel?: ProcessedKernel): string => {
  const normalizedType = irType.toLowerCase().trim();

  // Common programming languages (non-IR types)
  if (normalizedType === "python" || normalizedType.endsWith(".python")) {
    return "Python";
  }
  if (normalizedType === "json" || normalizedType.endsWith(".json")) {
    return "JSON";
  }

  // Try to get display name from kernel metadata (dynamic, no hardcoding)
  if (kernel?.metadata?.ir_stages) {
    const stageInfo = kernel.metadata.ir_stages.find(
      (stage: { name: string; extension: string; display_name?: string }) =>
        stage.name.toLowerCase() === normalizedType ||
        irType.toLowerCase().endsWith(stage.extension.toLowerCase())
    );
    if (stageInfo?.display_name) {
      return stageInfo.display_name;
    }
  }

  // Fallback: Generate reasonable display name for unknown IR types
  const baseType = normalizedType
    .replace(/^.*[\/\\]/, "")      // Remove path
    .replace(/\.(gz|xz|zst)$/, "")  // Remove compression extensions
    .replace(/^.*\./, "")           // Remove other extensions
    .replace(/[_-]/g, " ")          // Replace separators with spaces
    .trim();

  if (baseType) {
    // Convert to uppercase and add "IR" suffix if not present
    const upperName = baseType.toUpperCase();
    if (!upperName.endsWith("IR")) {
      return `${upperName} IR`;
    }
    return upperName;
  }

  // Final fallback to original input
  return irType;
};
