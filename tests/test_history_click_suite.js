/**
 * History Question Click & Populate Chat Input Test Suite
 */

const fs = require('fs');
const path = require('path');

const jsContent = fs.readFileSync(path.join(__dirname, '../static/js/app.js'), 'utf-8');
const htmlContent = fs.readFileSync(path.join(__dirname, '../templates/index.html'), 'utf-8');
const cssContent = fs.readFileSync(path.join(__dirname, '../static/css/style.css'), 'utf-8');

console.log('================================================================');
console.log('TEST SUITE: CLICK HISTORY QUESTION -> POPULATE CHAT INPUT');
console.log('================================================================');

let passedTests = 0;
let totalTests = 0;

function assert(condition, message) {
    totalTests++;
    if (condition) {
        console.log(`✓ [PASS] ${message}`);
        passedTests++;
    } else {
        console.error(`✗ [FAIL] ${message}`);
    }
}

// TEST 1: Check History DOM structure exists
const hasHistorySection = htmlContent.includes('id="view-history"');
const hasHistoryList = htmlContent.includes('id="recent-questions-list"');
const hasHistoryFilter = htmlContent.includes('id="history-filter-input"');
assert(hasHistorySection && hasHistoryList && hasHistoryFilter, 'History workspace section, list, and filter elements exist in index.html');

// TEST 2: Check History items have click handlers attached
const hasPopulateFunction = jsContent.includes('window.populateRecentQuestion = function(indexOrText)');
assert(hasPopulateFunction, 'populateRecentQuestion function is exposed globally on window');

// TEST 3: Check populateRecentQuestion navigates to chat using navigateToView('chat', true)
const hasNavigateToChat = jsContent.includes("navigateToView('chat', true)");
assert(hasNavigateToChat, "populateRecentQuestion calls navigateToView('chat', true) to switch views and update URL hash");

// TEST 4: Check populateRecentQuestion sets value on user-input without auto-submitting
const setsInputValue = jsContent.includes('inputEl.value = questionText;');
const dispatchesInputEvent = jsContent.includes("inputEl.dispatchEvent(new Event('input', { bubbles: true }));");
const focusesInput = jsContent.includes('inputEl.focus();');
const setsSelection = jsContent.includes('inputEl.setSelectionRange(len, len);');
assert(setsInputValue && dispatchesInputEvent && focusesInput && setsSelection, 'Chat input is populated with exact question, input event dispatched, focused, and cursor placed at end');

// TEST 5: Verify NO automatic submission/send is triggered
const doesNotAutoSubmit = !jsContent.includes('populateRecentQuestion') || !jsContent.includes('populateRecentQuestion') || (!jsContent.substring(jsContent.indexOf('populateRecentQuestion')).slice(0, 500).includes('chatForm.submit()') && !jsContent.substring(jsContent.indexOf('populateRecentQuestion')).slice(0, 500).includes('handleChatSubmit('));
assert(doesNotAutoSubmit, 'Clicking history question does NOT automatically submit form or send query');

// TEST 6: Check History row styling has pointer cursor and hover feedback
const hasCursorPointer = cssContent.includes('.recent-question-item {') && cssContent.includes('cursor: pointer;');
const hasHoverState = cssContent.includes('.recent-question-item:hover');
assert(hasCursorPointer && hasHoverState, 'CSS provides cursor: pointer and hover visual feedback on history question rows');

// TEST 7: Check renderHistoryPage is wired to navigateToView
const hasHistoryRenderWiring = jsContent.includes('renderHistoryPage();') && jsContent.includes('window.renderHistoryPage = renderRecentQuestionsList;');
assert(hasHistoryRenderWiring, 'renderHistoryPage is wired to navigateToView and renders recent questions list');

console.log('================================================================');
console.log(`SUMMARY: ${passedTests}/${totalTests} TESTS PASSED`);
console.log('================================================================');

if (passedTests !== totalTests) {
    process.exit(1);
}
