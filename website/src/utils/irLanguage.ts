/**
 * Utility functions for IR language display
 */

/**
 * Get a user-friendly display name for the IR language
 * @param irType - The type/name of IR file
 * @returns A human-readable language name
 */
export const getDisplayLanguage = (irType: string): string => {
  const normalized = irType.toLowerCase();

  const extMatch = normalized.match(/\.([^.]+)$/);
  const stageName = extMatch ? extMatch[1] : normalized;
  if (stageName === "python") {
    return "Python";
  }

  return stageName.replace(/_/g, " ").toUpperCase();
};
