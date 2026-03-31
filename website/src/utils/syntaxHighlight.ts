/**
 * Dynamic Syntax Highlight Detection
 *
 * This module provides utilities for dynamically determining the appropriate
 * syntax highlighter for different IR types, eliminating hardcoded mappings.
 */

import { ProcessedKernel } from './dataLoader';

/**
 * Infer syntax highlighter for an IR type based on naming patterns.
 *
 * This function uses heuristic rules to determine the appropriate highlighter
 * based on the IR type name and common conventions.
 *
 * @param irType - The IR type to determine highlighter for
 * @returns The syntax highlighter name
 *
 * @example
 * ```ts
 * inferSyntaxHighlighter('ttir'); // Returns 'mlir'
 * inferSyntaxHighlighter('ptx'); // Returns 'asm'
 * inferSyntaxHighlighter('llir'); // Returns 'llvm'
 * ```
 */
export function inferSyntaxHighlighter(irType: string): string {
    const lowerType = irType.toLowerCase();

    // MLIR dialects (TTIR, TTGIR, TTAdapter, BCMLIR, etc.)
    if (
        lowerType.includes('tt') ||      // TTIR, TTGIR, TTAdapter
        lowerType.includes('mlir') ||    // MLIR
        lowerType === 'ttadapter' ||
        lowerType === 'bcmlir'
    ) {
        return 'mlir';
    }

    // LLVM IR and variants
    if (
        lowerType.includes('ll') ||      // LLIR, LLVM
        lowerType === 'llvm'
    ) {
        return 'llvm';
    }

    // Assembly formats (PTX, AMDGCN, SASS, etc.)
    if (
        lowerType === 'ptx' ||
        lowerType === 'amdgcn' ||
        lowerType === 'sass' ||
        lowerType.includes('asm')
    ) {
        return 'asm';
    }

    // JSON format
    if (lowerType === 'json') {
        return 'json';
    }

    // Python format
    if (lowerType === 'py') {
        return 'python';
    }

    // Default to text
    return 'text';
}

/**
 * Select highlighter based on mapping_kind metadata (RFC Phase 2).
 *
 * This is the most accurate method, using the mapping_kind from ir_stages metadata
 * to determine the appropriate syntax highlighter.
 *
 * @param kernel - The processed kernel data
 * @param irType - The IR type
 * @returns The syntax highlighter name
 *
 * @example
 * ```ts
 * const kernel: ProcessedKernel = {
 *   metadata: {
 *     ir_stages: [
 *       { name: 'ptx', mapping_kind: 'ptx' },
 *       { name: 'ttir', mapping_kind: 'generic' }
 *     ]
 *   }
 * };
 *
 * selectHighlighterByMappingKind(kernel, 'ptx'); // Returns 'asm'
 * selectHighlighterByMappingKind(kernel, 'ttir'); // Returns 'mlir'
 * ```
 */
export function selectHighlighterByMappingKind(
    kernel: ProcessedKernel,
    irType: string
): string {
    if (!kernel.metadata?.ir_stages) {
        // Fallback to inference
        return inferSyntaxHighlighter(irType);
    }

    const stageInfo = kernel.metadata.ir_stages.find(
        stage => stage.name === irType
    );

    if (!stageInfo) {
        // Fallback to inference if stage not found
        return inferSyntaxHighlighter(irType);
    }

    // Select highlighter based on mapping_kind
    switch (stageInfo.mapping_kind) {
        case 'generic':
            // TTIR, TTGIR, TTAdapter, BCMLIR (MLIR dialects)
            return 'mlir';

        case 'ptx':
            // PTX, AMDGCN (assembly formats)
            return 'asm';

        case 'sass':
            // SASS (disassembly)
            return 'asm';

        case 'none':
            // No source mapping support
            return 'text';

        default:
            return 'text';
    }
}

/**
 * Smart highlighter selection (automatically chooses best method).
 *
 * This function detects whether ir_stages metadata is available and uses
 * the appropriate highlighter selection method.
 *
 * Priority:
 * 1. mapping_kind metadata (if available) - most accurate
 * 2. Heuristic inference (fallback)
 *
 * @param kernel - The processed kernel data
 * @param irType - The IR type
 * @returns The syntax highlighter name
 */
export function selectHighlighterSmart(
    kernel: ProcessedKernel,
    irType: string
): string {
    // Use metadata if available (RFC Phase 2)
    if (kernel.metadata?.ir_stages && kernel.metadata.ir_stages.length > 0) {
        return selectHighlighterByMappingKind(kernel, irType);
    }

    // Fallback to heuristic inference (Phase 1)
    return inferSyntaxHighlighter(irType);
}

/**
 * Extended language-to-highlighter mapping function.
 *
 * This is an enhanced version of the original mapLanguageToHighlighter
 * that uses dynamic detection instead of hardcoded mappings.
 *
 * @param language - The language/IR type string
 * @returns The syntax highlighter name
 *
 * @example
 * ```ts
 * mapLanguageToHighlighter('ttir'); // Returns 'mlir'
 * mapLanguageToHighlighter('ptx'); // Returns 'asm'
 * mapLanguageToHighlighter('python'); // Returns 'python'
 * ```
 */
export function mapLanguageToHighlighter(language: string): string {
    const lowerCaseLanguage = language.toLowerCase();

    // Explicit mappings for known languages
    const explicitMappings: Record<string, string> = {
        'python': 'python',
        'py': 'python',
        'cuda': 'cpp',
        'cpp': 'cpp',
        'c': 'cpp',
    };

    if (explicitMappings[lowerCaseLanguage]) {
        return explicitMappings[lowerCaseLanguage];
    }

    // Use dynamic inference for IR types
    return inferSyntaxHighlighter(lowerCaseLanguage);
}
