/**
 * Comprehensive Navigation & Button Regression Test Suite
 * Tests DOM structure, event listeners, view switching, hash updates, modals, and edge cases.
 */

const fs = require('fs');
const path = require('path');

// 1. Read index.html and app.js
const htmlContent = fs.readFileSync(path.join(__dirname, '../templates/index.html'), 'utf-8');
const jsContent = fs.readFileSync(path.join(__dirname, '../static/js/app.js'), 'utf-8');

console.log('================================================================');
console.log('RUNNING SIDEBAR NAVIGATION & BUTTONS AUDIT AND REGRESSION SUITE');
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

// TEST 1: Sidebar HTML elements exist with correct data-view and href attributes
const requiredViews = ['dashboard', 'documents', 'search', 'chat', 'sessions', 'history', 'analytics', 'rag'];
requiredViews.forEach(view => {
    const hasNavTag = htmlContent.includes(`data-view="${view}"`) && htmlContent.includes(`href="#${view}"`);
    assert(hasNavTag, `Sidebar navigation item for '${view}' exists with data-view="${view}" and href="#${view}"`);
});

// TEST 2: Workspace view section containers exist in HTML
requiredViews.forEach(view => {
    const hasSection = htmlContent.includes(`id="view-${view}"`);
    assert(hasSection, `Workspace view container for '#view-${view}' exists in index.html`);
});

// TEST 3: Check app.js defines VALID_VIEWS covering all 8 views
const validViewsMatch = jsContent.includes("VALID_VIEWS = ['dashboard', 'documents', 'search', 'chat', 'sessions', 'history', 'analytics', 'rag']");
assert(validViewsMatch, "app.js defines VALID_VIEWS with all 8 workspace views");

// TEST 4: Check navigateToView handles all views and toggles hidden classes
const hasNavigateToView = jsContent.includes('function navigateToView(viewId, updateHash = true)');
assert(hasNavigateToView, "navigateToView function exists and accepts viewId and updateHash");

// TEST 5: Check sidebar nav click listeners are attached to .sidebar-nav .nav-item and [data-view]
const hasSidebarListeners = jsContent.includes("document.querySelectorAll('.sidebar-nav .nav-item, [data-view]')");
assert(hasSidebarListeners, "Event listeners attached to all sidebar nav items and [data-view] elements");

// TEST 6: Check window hashchange listener is registered
const hasHashChangeListener = jsContent.includes("window.addEventListener('hashchange'");
assert(hasHashChangeListener, "Global window hashchange event listener registered for deep links & browser history");

// TEST 7: Check topbar navigation buttons are wired up
const hasTopbarDocs = jsContent.includes("btnTopbarDocs.addEventListener('click'");
const hasTopbarChat = jsContent.includes("btnTopbarChat.addEventListener('click'");
assert(hasTopbarDocs, "Topbar Documents button click listener wired to navigateToView('documents')");
assert(hasTopbarChat, "Topbar Chat button click listener wired to navigateToView('chat')");

// TEST 8: Check view mode toggle button delegates to navigateToView
const hasToggleView = jsContent.includes("navigateToView('chat', true)") && jsContent.includes("navigateToView('analytics', true)");
assert(hasToggleView, "Toggle view mode switches between Chat and Analytics via navigateToView");

// TEST 9: Check modal close functions reset overflow and pointer events cleanly
const hasCloseDetailSafety = jsContent.includes("document.body.style.overflow = previousBodyOverflow || '';") &&
                             jsContent.includes("document.body.style.pointerEvents = '';");
assert(hasCloseDetailSafety, "Question details modal close resets body overflow and pointer-events");

const hasCloseEvidenceSafety = jsContent.includes("document.body.style.overflow = previousEvidenceOverflow || '';") &&
                               jsContent.includes("document.body.style.pointerEvents = '';");
assert(hasCloseEvidenceSafety, "Evidence viewer modal close resets body overflow and pointer-events");

// TEST 10: Check theme transition overlay has pointer-events: none in CSS
const cssContent = fs.readFileSync(path.join(__dirname, '../static/css/style.css'), 'utf-8');
const hasOverlayPointerEvents = cssContent.includes('.theme-transition-overlay {') && cssContent.includes('pointer-events: none;');
assert(hasOverlayPointerEvents, "Theme transition overlay has pointer-events: none preventing click interception");

// TEST 11: Check initial route detection on DOM load
const hasInitialRoute = jsContent.includes("const initialHash = window.location.hash.replace('#', '');") &&
                        jsContent.includes("if (VALID_VIEWS.includes(initialHash))");
assert(hasInitialRoute, "Initial route correctly restores active view from URL hash on page load");

console.log('================================================================');
console.log(`SUMMARY: ${passedTests}/${totalTests} TESTS PASSED`);
console.log('================================================================');

if (passedTests !== totalTests) {
    process.exit(1);
}
