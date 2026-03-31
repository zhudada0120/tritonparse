/**
 * Dynamic IR Type Extraction Utilities
 *
 * This module provides utilities for dynamically extracting IR types from trace data,
 * eliminating hardcoded IR type lists in the frontend.
 *
 * This enables the frontend to automatically support any backend's IR types without
 * code modifications, achieving true backend-agnostic behavior.
 */

import { ProcessedKernel } from './dataLoader';

/**
 * Extract IR type from a filename.
 *
 * For example: "add_kernel.ptx" → "ptx", "kernel.ttadapter" → "ttadapter"
 *
 * @param filename - The filename to extract IR type from
 * @returns IR type string or null if not an IR file
 *
 * @example
 * ```ts
 * extractIRType("add_kernel.ptx"); // Returns "ptx"
 * extractIRType("kernel.ttadapter"); // Returns "ttadapter"
 * extractIRType("source.json"); // Returns null (not an IR file)
 * ```
 */
export function extractIRType(filename: string): string | null {
    if (!filename || !filename.includes('.')) {
        return null;
    }

    // Extract extension (last part after last dot)
    const parts = filename.split('.');
    const irType = parts[parts.length - 1];

    // Filter out non-IR file types
    const nonIRTypes = ['json', 'py', 'cpp', 'h', 'cu', 'source'];
    if (nonIRTypes.includes(irType)) {
        return null;
    }

    return irType;
}

/**
 * Extract all unique IR types from a ProcessedKernel.
 *
 * This function scans both irFiles and filePaths to extract IR types,
 * ensuring we don't miss any IR stages.
 *
 * @param kernel - The processed kernel data
 * @returns Sorted array of unique IR type strings
 *
 * @example
 * ```ts
 * const kernel: ProcessedKernel = {
 *   irFiles: {
 *     'kernel.ptx': '...',
 *     'kernel.ttir': '...',
 *     'source.json': '{}'
 *   }
 * };
 *
 * const types = extractAllIRTypes(kernel);
 * // Returns: ['ptx', 'ttir']
 * ```
 */
export function extractAllIRTypes(kernel: ProcessedKernel): string[] {
    const irTypes = new Set<string>();

    // Extract from irFiles
    if (kernel.irFiles) {
        Object.keys(kernel.irFiles).forEach(filename => {
            const irType = extractIRType(filename);
            if (irType) {
                irTypes.add(irType);
            }
        });
    }

    // Extract from filePaths (supplement)
    if (kernel.filePaths) {
        Object.keys(kernel.filePaths).forEach(filename => {
            const irType = extractIRType(filename);
            if (irType) {
                irTypes.add(irType);
            }
        });
    }

    // Return sorted array for consistent ordering
    return Array.from(irTypes).sort();
}

/**
 * Generate the source mapping property name for an IR type.
 *
 * For example: "ptx" → "ptx_lines", "ttir" → "ttir_lines"
 *
 * @param irType - The IR type
 * @returns The corresponding source mapping property name
 *
 * @example
 * ```ts
 * getMappingProperty('ptx'); // Returns "ptx_lines"
 * getMappingProperty('ttir'); // Returns "ttir_lines"
 * ```
 */
export function getMappingProperty(irType: string): string {
    return `${irType}_lines`;
}

/**
 * Dynamically generate IR types to check list for source mapping.
 *
 * This replaces the hardcoded irTypesToCheck array in CodeComparisonView.
 *
 * @param kernel - The processed kernel data
 * @returns Array of objects with type and property for source mapping lookup
 *
 * @example
 * ```ts
 * const kernel: ProcessedKernel = { ... };
 * const typesToCheck = generateIRTypesToCheck(kernel);
 * // Returns:
 * // [
 * //   { type: "ptx", property: "ptx_lines" },
 * //   { type: "ttir", property: "ttir_lines" },
 * //   ...
 * // ]
 * ```
 */
export function generateIRTypesToCheck(
    kernel: ProcessedKernel
): Array<{ type: string; property: string }> {
    if (!kernel.metadata?.ir_stages) {
        // Fallback: extract from filenames
        return extractAllIRTypes(kernel).map(irType => ({
            type: irType,
            property: getMappingProperty(irType)
        }));
    }

    // Use ir_stages metadata (RFC design)
    return kernel.metadata.ir_stages
        .filter(stage => stage.is_text && stage.supports_source_mapping)
        .map(stage => ({
            type: stage.name,
            property: getMappingProperty(stage.name)
        }));
}

/**
 * Extract IR types from metadata (RFC Phase 2 implementation).
 *
 * If the trace contains ir_stages metadata (from RFC backend implementation),
 * this function extracts IR types with their capability information.
 *
 * @param kernel - The processed kernel data
 * @returns Array of IR type strings that support source mapping
 *
 * @example
 * ```ts
 * const kernel: ProcessedKernel = {
 *   metadata: {
 *     ir_stages: [
 *       { name: "ttir", is_text: true, supports_source_mapping: true },
 *       { name: "ptx", is_text: true, supports_source_mapping: true },
 *       { name: "cubin", is_text: false, supports_source_mapping: false }
 *     ]
 *   }
 * };
 *
 * const types = extractIRTypesFromMetadata(kernel);
 * // Returns: ['ptx', 'ttir'] (only text IRs with source mapping support)
 * ```
 */
export function extractIRTypesFromMetadata(kernel: ProcessedKernel): string[] {
    if (!kernel.metadata?.ir_stages) {
        // Fallback to filename-based extraction
        return extractAllIRTypes(kernel);
    }

    return kernel.metadata.ir_stages
        .filter(stage => stage.is_text && stage.supports_source_mapping)
        .map(stage => stage.name)
        .sort();
}

/**
 * Smart IR type extraction (automatically chooses best method).
 *
 * This function automatically detects whether the trace has ir_stages metadata
 * and uses the appropriate extraction method.
 *
 * Priority:
 * 1. ir_stages metadata (if available) - most accurate
 * 2. Filename-based extraction (fallback)
 *
 * @param kernel - The processed kernel data
 * @returns Array of IR type strings
 */
export function extractIRTypesSmart(kernel: ProcessedKernel): string[] {
    // Use metadata if available (RFC Phase 2)
    if (kernel.metadata?.ir_stages && kernel.metadata.ir_stages.length > 0) {
        return extractIRTypesFromMetadata(kernel);
    }

    // Fallback to filename-based extraction (Phase 1)
    return extractAllIRTypes(kernel);
}

/**
 * Check if a kernel has a specific IR type.
 *
 * @param kernel - The processed kernel data
 * @param irType - The IR type to check for
 * @returns True if the kernel has this IR type
 *
 * @example
 * ```ts
 * const kernel: ProcessedKernel = {
 *   irFiles: { 'kernel.ptx': '...' }
 * };
 *
 * hasIRType(kernel, 'ptx'); // Returns true
 * hasIRType(kernel, 'ttir'); // Returns false
 * ```
 */
export function hasIRType(kernel: ProcessedKernel, irType: string): boolean {
    return extractIRTypesSmart(kernel).includes(irType);
}

/**
 * Get the file content for a specific IR type.
 *
 * This searches through irFiles to find the content for the given IR type.
 *
 * @param kernel - The processed kernel data
 * @param irType - The IR type to get content for
 * @returns The IR file content or undefined if not found
 *
 * @example
 * ```ts
 * const kernel: ProcessedKernel = {
 *   irFiles: {
 *     'add_kernel.ptx': '.version 6.4 ...',
 *     'add_kernel.ttir': 'module { ... }'
 *   }
 * };
 *
 * const ptxContent = getIRFileContent(kernel, 'ptx');
 * // Returns: '.version 6.4 ...'
 * ```
 */
export function getIRFileContent(
    kernel: ProcessedKernel,
    irType: string
): string | undefined {
    if (!kernel.irFiles) {
        return undefined;
    }

    // Find the file that matches this IR type
    const filename = Object.keys(kernel.irFiles).find(fname => {
        const type = extractIRType(fname);
        return type === irType;
    });

    return filename ? kernel.irFiles[filename] : undefined;
}

/**
 * Get the file path for a specific IR type.
 *
 * This searches through filePaths to find the path for the given IR type.
 *
 * @param kernel - The processed kernel data
 * @param irType - The IR type to get path for
 * @returns The IR file path or undefined if not found
 */
export function getIRFilePath(
    kernel: ProcessedKernel,
    irType: string
): string | undefined {
    if (!kernel.filePaths) {
        return undefined;
    }

    // Find the file that matches this IR type
    const filename = Object.keys(kernel.filePaths).find(fname => {
        const type = extractIRType(fname);
        return type === irType;
    });

    return filename ? kernel.filePaths[filename] : undefined;
}

/**
 * Generate IR file options for UI selectors.
 *
 * This creates user-friendly options for IR file selector components.
 *
 * @param kernel - The processed kernel data
 * @returns Array of option objects with value and label
 *
 * @example
 * ```ts
 * const kernel: ProcessedKernel = { ... };
 * const options = generateIRFileOptions(kernel);
 * // Returns:
 * // [
 * //   { value: "ptx", label: "PTX IR" },
 * //   { value: "ttir", label: "TTIR" },
 * //   { value: "ttadapter", label: "TTADAPTER IR" },
 * //   ...
 * // ]
 * ```
 */
export function generateIRFileOptions(
    kernel: ProcessedKernel
): Array<{ value: string; label: string }> {
    const irTypes = extractIRTypesSmart(kernel);

    return irTypes.map(irType => ({
        value: irType,
        label: generateIRDisplayName(irType)
    }));
}

/**
 * Generate a human-readable display name for an IR type.
 *
 * @param irType - The IR type
 * @returns Human-readable display name using simple rules
 *
 * @example
 * ```ts
 * generateIRDisplayName('ptx'); // Returns "PTX IR"
 * generateIRDisplayName('ttir'); // Returns "TTIR" (already ends with IR)
 * generateIRDisplayName('xyz'); // Returns "XYZ IR" (rule-based for any type)
 * ```
 */
export function generateIRDisplayName(irType: string): string {
    // Rule-based display name generation (zero hardcoding)
    const upperType = irType.toUpperCase();

    // Simple rule: add "IR" suffix if not already present
    if (upperType.endsWith("IR")) {
        return upperType;
    }
    return `${upperType} IR`;
}
