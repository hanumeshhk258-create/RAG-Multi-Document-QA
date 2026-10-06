/**
 * tests/test_frontend_error_handling.js
 * Unit tests for frontend error handling logic in static/js/app.js.
 */

const fs = require('fs');
const path = require('path');

const appJsPath = path.join(__dirname, '..', 'static', 'js', 'app.js');
const code = fs.readFileSync(appJsPath, 'utf8');

// 1. Verify mapErrorCodeToMessage logic directly from app.js
const mapMatch = code.match(/function mapErrorCodeToMessage\([\s\S]*?\n    \}/);
if (!mapMatch) {
    throw new Error('mapErrorCodeToMessage not found in app.js');
}

const mapFnStr = mapMatch[0];
const mapErrorCodeToMessage = new Function(
    'errorCode', 
    'defaultMsg', 
    mapFnStr.slice(mapFnStr.indexOf('{') + 1, mapFnStr.lastIndexOf('}'))
);

console.log('--- TEST SUITE 1: mapErrorCodeToMessage ---');

// Test 1A: Backend error message prioritized over VALIDATION_ERROR code
const msg1 = mapErrorCodeToMessage('VALIDATION_ERROR', "No indexed documents are available. Please click 'Index Documents' first.");
console.log('1A: Result =', msg1);
if (msg1 !== "No indexed documents are available. Please click 'Index Documents' first.") {
    throw new Error('Test 1A failed');
}

// Test 1B: Backend custom error prioritized over INTERNAL_SERVER_ERROR code
const msg2 = mapErrorCodeToMessage('INTERNAL_SERVER_ERROR', "Database disk quota exceeded.");
console.log('1B: Result =', msg2);
if (msg2 !== "Database disk quota exceeded.") {
    throw new Error('Test 1B failed');
}

// Test 1C: Backend custom error prioritized over AI_AUTH_ERROR code
const msg3 = mapErrorCodeToMessage('AI_AUTH_ERROR', "Gemini API key is invalid or expired.");
console.log('1C: Result =', msg3);
if (msg3 !== "Gemini API key is invalid or expired.") {
    throw new Error('Test 1C failed');
}

// Test 1D: Error code only when error message is null/empty
const msg4 = mapErrorCodeToMessage('AI_RATE_LIMIT', null);
console.log('1D: Result =', msg4);
if (msg4 !== "AI request limit reached. Please try again shortly.") {
    throw new Error('Test 1D failed');
}

// Test 1E: Error code only for AI_TIMEOUT
const msg5 = mapErrorCodeToMessage('AI_TIMEOUT', '');
console.log('1E: Result =', msg5);
if (msg5 !== "AI response timed out. Please try again.") {
    throw new Error('Test 1E failed');
}

// Test 1F: Neither code nor message
const msg6 = mapErrorCodeToMessage(null, null);
console.log('1F: Result =', msg6);
if (msg6 !== "The AI service is temporarily unavailable. Please try again.") {
    throw new Error('Test 1F failed');
}

console.log('\n--- TEST SUITE 2: Response Extraction Logic ---');

// Simulate the logic in app.js for non-ok response parsing
function simulateResponseExtraction(status, errData, textBody = '') {
    let errorMsg = null;
    let errorCode = null;

    if (errData && typeof errData === 'object') {
        errorMsg = errData.error || errData.message || null;
        errorCode = errData.error_code || null;
    }

    const finalMsg = errorMsg
        ? mapErrorCodeToMessage(errorCode, errorMsg)
        : (errorCode
            ? mapErrorCodeToMessage(errorCode, null)
            : (status === 400
                ? "Invalid request. Please check your query and document selection."
                : (status >= 500
                    ? `Server error (HTTP ${status}). Please check the backend logs.`
                    : `Request failed with status ${status}.`)));

    return finalMsg;
}

// Test 2A: 400 with backend JSON error
const res2A = simulateResponseExtraction(400, {
    success: false,
    error: "No indexed documents are available. Please click 'Index Documents' first.",
    error_code: "VALIDATION_ERROR"
});
console.log('2A (400 JSON):', res2A);
if (res2A !== "No indexed documents are available. Please click 'Index Documents' first.") {
    throw new Error('Test 2A failed');
}

// Test 2B: 400 with generic/empty error
const res2B = simulateResponseExtraction(400, null);
console.log('2B (400 non-JSON):', res2B);
if (res2B !== "Invalid request. Please check your query and document selection.") {
    throw new Error('Test 2B failed');
}

// Test 2C: 500 without JSON (e.g. HTML error page)
const res2C = simulateResponseExtraction(500, null);
console.log('2C (500 HTML):', res2C);
if (res2C !== "Server error (HTTP 500). Please check the backend logs.") {
    throw new Error('Test 2C failed');
}

console.log('\n--- TEST SUITE 3: Catch Block & Fallback Verification ---');
const chatCatchMatch = code.match(/console\.error\('Chat error:', err\);([\s\S]*?)\} finally/);
if (!chatCatchMatch) {
    throw new Error('sendMessage catch block not found');
}
const catchBody = chatCatchMatch[1];
if (!catchBody.includes("'The AI service is temporarily unavailable. Please try again.'")) {
    throw new Error("catch block does not have fallback message for network failures");
}
if (!catchBody.includes("err.name === 'AbortError'")) {
    throw new Error("catch block does not handle AbortError");
}
console.log('3: Catch block properly preserves generic fallback for network/connection failures and timeout handling.');

console.log('\n========================================');
console.log('ALL FRONTEND ERROR HANDLING TESTS PASSED');
console.log('========================================');
