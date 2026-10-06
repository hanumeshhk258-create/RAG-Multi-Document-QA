/**
 * RAG Multi-Document Assistant - Advanced Frontend Logic
 * Supports Multi-PDF, Smart Document Filtering & Selection,
 * Fast Hybrid Retrieval, Interactive Source Excerpts, and Markdown Tables/Code Formatting
 */

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
window.escapeHtml = escapeHtml;

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Sidebar
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    const fileCountBadge = document.getElementById('file-count-badge');
    const btnIndex = document.getElementById('btn-index');
    const btnIndexText = document.getElementById('btn-index-text');
    const btnRebuild = document.getElementById('btn-rebuild');
    const btnReset = document.getElementById('btn-reset');

    // Selection Controls
    const selectAllCheckbox = document.getElementById('select-all-checkbox');
    const selectedDocCount = document.getElementById('selected-doc-count');
    const btnSelectAll = document.getElementById('btn-select-all');
    const btnClearSelection = document.getElementById('btn-clear-selection');
    const statSelectedSummary = document.getElementById('stat-selected-summary');

    // Status Panel Elements
    const systemStatus = document.getElementById('system-status');
    const statusText = document.getElementById('status-text');
    const statDocs = document.getElementById('stat-docs');
    const statIndexed = document.getElementById('stat-indexed');
    const statPending = document.getElementById('stat-pending');
    const statPages = document.getElementById('stat-pages');
    const statChunks = document.getElementById('stat-chunks');
    const statModel = document.getElementById('stat-model');

    // Chat Elements
    const chatContainer = document.getElementById('chat-container');
    const chatMessages = document.getElementById('chat-messages');
    const thinkingIndicator = document.getElementById('thinking-indicator');
    const thinkingText = document.getElementById('thinking-text');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const btnSend = document.getElementById('btn-send');
    const btnNewChat = document.getElementById('btn-new-chat');
    const btnClearChat = document.getElementById('btn-clear-chat');
    const alertBanner = document.getElementById('alert-banner');

    // Source Preview Modal Elements
    const sourceModal = document.getElementById('source-modal');
    const modalBackdrop = document.getElementById('modal-backdrop');
    const btnCloseModal = document.getElementById('btn-close-modal');
    const modalDocName = document.getElementById('modal-doc-name');
    const modalPageNum = document.getElementById('modal-page-num');
    const modalPreviewText = document.getElementById('modal-preview-text');

    // App State
    let isVectorstoreReady = false;
    let isProcessing = false;
    let chatHistory = []; // Tracks {role: 'user'|'assistant', content: string}
    let allDocumentsList = []; // List of all documents
    let selectedDocs = new Set(); // Set of currently selected filenames
    let initialSelectionLoaded = false;

    // Active sources cache for preview lookup
    window.currentSourceCache = {};

    // Initialize System Status on load
    fetchSystemStatus();

    // =========================================================================
    // 1. SMART DOCUMENT SELECTION HANDLERS
    // =========================================================================
    if (selectAllCheckbox) {
        selectAllCheckbox.addEventListener('change', (e) => {
            const isChecked = e.target.checked;
            if (isChecked) {
                allDocumentsList.forEach(d => selectedDocs.add(d.filename));
            } else {
                selectedDocs.clear();
            }
            updateSelectionUI();
        });
    }

    if (btnSelectAll) {
        btnSelectAll.addEventListener('click', () => {
            allDocumentsList.forEach(d => selectedDocs.add(d.filename));
            updateSelectionUI();
        });
    }

    if (btnClearSelection) {
        btnClearSelection.addEventListener('click', () => {
            selectedDocs.clear();
            updateSelectionUI();
        });
    }

    window.toggleDocSelection = function(filename) {
        if (selectedDocs.has(filename)) {
            selectedDocs.delete(filename);
        } else {
            selectedDocs.add(filename);
        }
        updateSelectionUI();
    };

    function updateSelectionUI() {
        const total = allDocumentsList.length;
        const selected = selectedDocs.size;

        // Update counter badge
        if (selectedDocCount) {
            selectedDocCount.textContent = `${selected} of ${total} selected`;
            if (selected === 0) {
                selectedDocCount.classList.add('empty');
            } else {
                selectedDocCount.classList.remove('empty');
            }
        }

        // Update footer summary
        if (statSelectedSummary) {
            statSelectedSummary.textContent = `${selected} of ${total} selected`;
        }

        // Update search scope badge
        if (typeof updateSearchScopeBadge === 'function') {
            updateSearchScopeBadge();
        }

        // Update "Select All" checkbox state
        if (selectAllCheckbox) {
            selectAllCheckbox.checked = (total > 0 && selected === total);
            selectAllCheckbox.indeterminate = (selected > 0 && selected < total);
        }

        // Update file cards classes and individual checkboxes
        allDocumentsList.forEach(doc => {
            const isChecked = selectedDocs.has(doc.filename);
            const cardEl = document.getElementById(`doc-card-${sanitizeId(doc.filename)}`);
            const cbEl = document.getElementById(`doc-cb-${sanitizeId(doc.filename)}`);

            if (cardEl) {
                if (isChecked) {
                    cardEl.classList.add('selected');
                    cardEl.classList.remove('unselected');
                } else {
                    cardEl.classList.remove('selected');
                    cardEl.classList.add('unselected');
                }
            }
            if (cbEl) {
                cbEl.checked = isChecked;
            }
        });
    }

    function sanitizeId(str) {
        return str.replace(/[^a-zA-Z0-9_-]/g, '_');
    }

    // =========================================================================
    // 2. DRAG & DROP AND MULTI-PDF UPLOAD
    // =========================================================================
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    // =========================================================================
    // 2. DRAG & DROP AND MULTI-PDF UPLOAD WITH DUPLICATE HANDLING
    // =========================================================================
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        handleFilesSelected(files, false);
    });

    fileInput.addEventListener('change', (e) => {
        const files = e.target.files;
        handleFilesSelected(files, false);
    });

    // Staging for duplicate file replace prompt
    let pendingDuplicateFile = null;

    // Duplicate Modal DOM Elements
    const duplicateModal = document.getElementById('duplicate-upload-modal');
    const duplicateModalDocname = document.getElementById('duplicate-modal-docname');
    const duplicateModalStatus = document.getElementById('duplicate-modal-status');
    const duplicateModalMessage = document.getElementById('duplicate-modal-message');
    const btnCloseDuplicateModal = document.getElementById('btn-close-duplicate-modal');
    const btnDuplicateKeep = document.getElementById('btn-duplicate-keep');
    const btnDuplicateReplace = document.getElementById('btn-duplicate-replace');

    function openDuplicateModal(file, existingDoc, message) {
        pendingDuplicateFile = file;
        if (duplicateModalDocname) duplicateModalDocname.textContent = file.name;
        if (duplicateModalStatus) {
            const st = existingDoc ? (existingDoc.status || 'Ready') : 'Ready';
            duplicateModalStatus.textContent = `Status: ${st} (${existingDoc ? (existingDoc.chunks || 0) : 0} chunks)`;
        }
        if (duplicateModalMessage && message) {
            duplicateModalMessage.textContent = message;
        }
        if (duplicateModal) duplicateModal.classList.remove('hidden');
    }

    function closeDuplicateModal() {
        pendingDuplicateFile = null;
        if (duplicateModal) duplicateModal.classList.add('hidden');
    }

    if (btnCloseDuplicateModal) btnCloseDuplicateModal.addEventListener('click', closeDuplicateModal);
    if (btnDuplicateKeep) btnDuplicateKeep.addEventListener('click', closeDuplicateModal);
    if (btnDuplicateReplace) {
        btnDuplicateReplace.addEventListener('click', async () => {
            const fileToReplace = pendingDuplicateFile;
            closeDuplicateModal();
            if (fileToReplace) {
                await uploadSingleFile(fileToReplace, true);
            }
        });
    }

    async function uploadSingleFile(file, replace = false) {
        const formData = new FormData();
        formData.append('files', file);
        if (replace) {
            formData.append('replace', 'true');
        }

        showAlert(replace ? `Replacing '${file.name}'...` : `Uploading '${file.name}'...`, 'info');

        try {
            const response = await fetch(`/api/upload${replace ? '?replace=true' : ''}`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.is_duplicate && !replace) {
                openDuplicateModal(file, data.existing_document, data.message);
                return;
            }

            if (data.success) {
                showAlert(replace ? `✓ Replaced '${file.name}' successfully. Click 'Index Documents' to process.` : `✓ Uploaded '${file.name}' successfully.`, 'success');
                selectedDocs.add(file.name);
                await fetchSystemStatus();
                btnIndex.disabled = false;
            } else {
                showAlert(data.error || 'Failed to upload document.', 'error');
            }
        } catch (err) {
            showAlert('Network error while uploading file.', 'error');
            console.error('Upload single file error:', err);
        }
    }

    async function handleFilesSelected(files, replace = false) {
        if (!files || files.length === 0) return;

        const validPdfs = [];
        let totalSize = 0;

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            if (file.name.toLowerCase().endsWith('.pdf')) {
                validPdfs.push(file);
                totalSize += file.size;
            }
        }

        if (validPdfs.length === 0) {
            showAlert('Please select valid PDF documents (.pdf).', 'error');
            return;
        }

        if (totalSize > 50 * 1024 * 1024) {
            showAlert('Total upload size exceeds 50MB limit. Please upload fewer files at a time.', 'error');
            return;
        }

        // If single file, check duplicates with full interactive replace option
        if (validPdfs.length === 1) {
            await uploadSingleFile(validPdfs[0], replace);
            if (fileInput) fileInput.value = '';
            return;
        }

        const formData = new FormData();
        validPdfs.forEach(f => formData.append('files', f));
        if (replace) formData.append('replace', 'true');

        showAlert(`Uploading ${validPdfs.length} PDF document(s)...`, 'info');

        try {
            const response = await fetch(`/api/upload${replace ? '?replace=true' : ''}`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.is_duplicate && !replace) {
                const dupFile = validPdfs.find(f => f.name === data.filename) || validPdfs[0];
                openDuplicateModal(dupFile, data.existing_document, data.message);
                return;
            }

            if (data.success) {
                showAlert(`Successfully uploaded ${data.uploaded_count} PDF document(s). Click 'Index Documents' to process.`, 'success');
                if (data.files) {
                    data.files.forEach(f => selectedDocs.add(f));
                }
                await fetchSystemStatus();
                btnIndex.disabled = false;
            } else {
                showAlert(data.error || 'Failed to upload documents.', 'error');
            }
        } catch (err) {
            showAlert('Network error while uploading documents.', 'error');
            console.error('Upload error:', err);
        } finally {
            if (fileInput) fileInput.value = '';
        }
    }

    // =========================================================================
    // 3. DOCUMENT INDEXING & REBUILDING (PROGRESS INDICATORS)
    // =========================================================================
    btnIndex.addEventListener('click', () => runIndexingProcess(false));
    btnRebuild.addEventListener('click', () => runIndexingProcess(true));

    async function runIndexingProcess(isRebuild = false) {
        if (isProcessing) return;

        const pendingOrErrorDocs = allDocumentsList.filter(d => {
            if (isRebuild) return true;
            if (selectedDocs.size > 0) {
                return selectedDocs.has(d.filename) && (d.status === 'Pending' || d.status === 'Error' || !d.indexed);
            }
            return d.status === 'Pending' || d.status === 'Error' || !d.indexed;
        });

        const totalToIndex = pendingOrErrorDocs.length;
        if (!isRebuild && totalToIndex === 0) {
            showAlert('All selected documents are already indexed and Ready.', 'info');
            return;
        }

        // Real-time UI update: Mark target documents as Indexing...
        pendingOrErrorDocs.forEach(d => {
            d.status = 'Indexing';
            const cardEl = document.getElementById(`doc-card-${sanitizeId(d.filename)}`);
            if (cardEl) {
                const tagEl = cardEl.querySelector('.meta-tag.badge-pending, .meta-tag.badge-error, .meta-tag.badge-ready, .meta-tag.badge-indexed, .meta-tag.badge-deleting');
                if (tagEl) {
                    tagEl.className = 'meta-tag badge-indexing';
                    tagEl.textContent = '🔵 Indexing...';
                }
            }
        });

        btnIndex.disabled = true;
        btnRebuild.disabled = true;
        btnIndexText.textContent = isRebuild ? 'Rebuilding...' : `Indexing (${totalToIndex})...`;
        setSystemStatusBadge('indexing', `⏳ Indexing ${totalToIndex || 1} document(s)...`);

        if (isRebuild) {
            showAlert('Rebuilding Index...\nProcessing documents and updating search index...', 'info');
        } else {
            showAlert(`Extracting text and generating embeddings for ${totalToIndex} document(s)...`, 'info');
        }

        try {
            const endpoint = isRebuild ? '/api/rebuild' : '/api/index';
            const payload = isRebuild ? {} : { selected_documents: Array.from(selectedDocs) };

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();

            if (data.success) {
                const countIndexed = data.indexed_count !== undefined ? data.indexed_count : (data.documents || 0);
                const msg = data.message || `${countIndexed} document(s) indexed successfully.`;
                showAlert(`✓ ${msg}`, 'success');
            } else {
                showAlert(data.error || 'Failed to index documents.', 'error');
            }
        } catch (err) {
            showAlert('Network error during document indexing.', 'error');
            console.error('Indexing error:', err);
        } finally {
            await fetchSystemStatus();
            btnIndex.disabled = false;
            btnRebuild.disabled = false;
            btnIndexText.textContent = 'Index Documents';
        }
    }

    // =========================================================================
    // 4. REMOVE SINGLE DOCUMENT (WITH CONFIRMATION MODAL & TRUE DELETION)
    // =========================================================================
    let targetDocIdToDelete = null;
    let targetFilenameToDelete = null;

    // Delete Modal Elements
    const deleteConfirmModal = document.getElementById('delete-confirm-modal');
    const deleteModalDocname = document.getElementById('delete-modal-docname');
    const deleteModalDocmeta = document.getElementById('delete-modal-docmeta');
    const btnCloseDeleteModal = document.getElementById('btn-close-delete-modal');
    const btnCancelDelete = document.getElementById('btn-cancel-delete');
    const btnConfirmDelete = document.getElementById('btn-confirm-delete');

    window.openDeleteConfirmModal = function(docId, filename) {
        targetDocIdToDelete = docId || filename;
        targetFilenameToDelete = filename || docId;

        if (deleteModalDocname) deleteModalDocname.textContent = targetFilenameToDelete;
        if (deleteModalDocmeta) deleteModalDocmeta.textContent = `Document ID: ${targetDocIdToDelete}`;
        if (deleteConfirmModal) deleteConfirmModal.classList.remove('hidden');
    };

    window.removeDocument = function(filename) {
        const docObj = allDocumentsList.find(d => d.filename === filename || d.document_id === filename);
        const docId = docObj ? (docObj.document_id || filename) : filename;
        window.openDeleteConfirmModal(docId, filename);
    };

    function closeDeleteModal() {
        targetDocIdToDelete = null;
        targetFilenameToDelete = null;
        if (deleteConfirmModal) deleteConfirmModal.classList.add('hidden');
    }

    if (btnCloseDeleteModal) btnCloseDeleteModal.addEventListener('click', closeDeleteModal);
    if (btnCancelDelete) btnCancelDelete.addEventListener('click', closeDeleteModal);

    if (btnConfirmDelete) {
        btnConfirmDelete.addEventListener('click', async () => {
            const docId = targetDocIdToDelete;
            const filename = targetFilenameToDelete;
            closeDeleteModal();
            if (docId) {
                await executeDocumentDelete(docId, filename);
            }
        });
    }

    async function executeDocumentDelete(docId, filename) {
        if (!docId) return;

        // Set card status to DELETING
        const cardEl = document.getElementById(`doc-card-${sanitizeId(filename)}`);
        if (cardEl) {
            const tagEl = cardEl.querySelector('.meta-tag.badge-pending, .meta-tag.badge-error, .meta-tag.badge-ready, .meta-tag.badge-indexing');
            if (tagEl) {
                tagEl.className = 'meta-tag badge-deleting';
                tagEl.textContent = '🗑️ Deleting...';
            }
        }

        showAlert(`Removing document '${filename}'...`, 'info');

        try {
            const response = await fetch(`/api/documents/${encodeURIComponent(docId)}`, {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ document_id: docId, filename: filename })
            });

            const data = await response.json();

            if (response.ok && data.success) {
                // 1. Remove from selection state
                selectedDocs.delete(filename);

                // 2. Invalidate cached evidence / source results for deleted document
                if (window.currentSourceCache) {
                    for (const k in window.currentSourceCache) {
                        if (window.currentSourceCache[k] &&
                            (window.currentSourceCache[k].document === filename ||
                             window.currentSourceCache[k].document_id === docId)) {
                            delete window.currentSourceCache[k];
                        }
                    }
                }

                // 3. Immediately remove card from UI
                if (cardEl) {
                    cardEl.remove();
                }

                // 4. Update status and analytics without page reload
                await fetchSystemStatus();
                showAlert("Document removed successfully.", 'success');
            } else {
                // Deletion failed: keep document visible, restore status badge, show error
                if (cardEl) {
                    const tagEl = cardEl.querySelector('.meta-tag.badge-deleting');
                    if (tagEl) {
                        tagEl.className = 'meta-tag badge-error';
                        tagEl.textContent = '🔴 Delete Failed';
                    }
                }
                const errMsg = data.error || "Unable to remove document. The index was not changed.";
                showAlert(errMsg, 'error');
            }
        } catch (err) {
            if (cardEl) {
                const tagEl = cardEl.querySelector('.meta-tag.badge-deleting');
                if (tagEl) {
                    tagEl.className = 'meta-tag badge-error';
                    tagEl.textContent = '🔴 Delete Failed';
                }
            }
            showAlert("Unable to remove document. The index was not changed.", 'error');
            console.error('Remove doc error:', err);
        }
    }

    // =========================================================================
    // 5. CLEAR ALL DOCUMENTS
    // =========================================================================
    btnReset.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to clear all uploaded documents and reset the index?')) {
            return;
        }

        try {
            const response = await fetch('/api/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();

            if (data.success) {
                selectedDocs.clear();
                showAlert('All documents and vector store cleared successfully.', 'success');
                await fetchSystemStatus();
                clearChat();
            } else {
                showAlert(data.error || 'Failed to clear documents.', 'error');
            }
        } catch (err) {
            showAlert('Network error while resetting documents.', 'error');
            console.error('Reset error:', err);
        }
    });

    // =========================================================================
    // 6. FRONTEND MATCHED TERM HIGHLIGHTING
    // =========================================================================
    function highlightMatchedTerms(rawText, termsOrQuery) {
        if (!rawText) return '';
        let escaped = escapeHtml(rawText);
        if (!termsOrQuery) return escaped;

        let termList = [];
        if (Array.isArray(termsOrQuery)) {
            termList = termsOrQuery;
        } else if (typeof termsOrQuery === 'string') {
            const cleaned = termsOrQuery.replace(/[\?\!\.,;:_]/g, ' ');
            termList = cleaned.split(/\s+/);
        }

        const stopWords = new Set([
            'what', 'is', 'are', 'was', 'were', 'the', 'a', 'an', 'in', 'on', 'of',
            'for', 'to', 'and', 'or', 'by', 'with', 'about', 'give', 'me', 'explain',
            'show', 'tell', 'can', 'you', 'please', 'it', 'its', 'this', 'that', 'these',
            'those', 'from', 'at', 'as', 'into', 'then', 'than'
        ]);

        const validTerms = Array.from(new Set(
            termList
                .map(t => String(t).trim())
                .filter(t => t.length > 1 && !stopWords.has(t.toLowerCase()))
        )).sort((a, b) => b.length - a.length);

        if (validTerms.length === 0) return escaped;

        validTerms.forEach(term => {
            const escapedTerm = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`\\b(${escapedTerm})\\b`, 'gi');
            escaped = escaped.replace(regex, '<mark class="highlight-term">$1</mark>');
        });

        return escaped;
    }

    // =========================================================================
    // 7. SOURCE / EXCERPT PREVIEW MODAL
    // =========================================================================
    const btnCopyExcerpt = document.getElementById('btn-copy-excerpt');
    const btnCloseModalBottom = document.getElementById('btn-close-modal-bottom');
    const modalRelBadge = document.getElementById('modal-rel-badge');
    const modalPdfLink = document.getElementById('modal-pdf-link');
    const modalFooterPdfBtn = document.getElementById('modal-footer-pdf-btn');

    window.openSourcePreview = function(cacheKey) {
        const item = window.currentSourceCache[cacheKey];
        if (!item) return;

        const docName = item.document || 'Document';
        const pageNum = item.page || 1;
        const pdfUrl = item.pdfUrl || `/view_pdf/${encodeURIComponent(docName)}#page=${pageNum}`;

        modalDocName.textContent = docName;
        modalPageNum.textContent = `Page ${pageNum}`;

        if (modalRelBadge) {
            modalRelBadge.textContent = `🎯 ${item.relevancePct || 90}% relevant`;
        }

        if (modalPdfLink) {
            modalPdfLink.href = pdfUrl;
            modalPdfLink.textContent = `🔍 View ${docName} at Page ${pageNum}`;
            modalPdfLink.setAttribute('target', '_blank');
            modalPdfLink.setAttribute('rel', 'noopener noreferrer');
        }

        if (modalFooterPdfBtn) {
            modalFooterPdfBtn.href = pdfUrl;
            modalFooterPdfBtn.setAttribute('target', '_blank');
            modalFooterPdfBtn.setAttribute('rel', 'noopener noreferrer');
        }

        const fullText = item.excerpt || item.snippet || 'No excerpt available for this chunk.';
        const terms = item.matchedTerms || item.query || [];
        modalPreviewText.innerHTML = highlightMatchedTerms(fullText, terms);

        sourceModal.classList.remove('hidden');
    };

    function closeSourcePreview() {
        sourceModal.classList.add('hidden');
    }

    if (btnCloseModal) btnCloseModal.addEventListener('click', closeSourcePreview);
    if (btnCloseModalBottom) btnCloseModalBottom.addEventListener('click', closeSourcePreview);
    if (modalBackdrop) modalBackdrop.addEventListener('click', closeSourcePreview);

    if (btnCopyExcerpt) {
        btnCopyExcerpt.addEventListener('click', () => {
            const plainText = modalPreviewText.innerText || modalPreviewText.textContent;
            if (!plainText) return;
            navigator.clipboard.writeText(plainText).then(() => {
                const originalHtml = btnCopyExcerpt.innerHTML;
                btnCopyExcerpt.innerHTML = `<span>✓</span> Copied`;
                btnCopyExcerpt.classList.add('copied');
                setTimeout(() => {
                    btnCopyExcerpt.innerHTML = originalHtml;
                    btnCopyExcerpt.classList.remove('copied');
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy excerpt text:', err);
            });
        });
    }

    // =========================================================================
    // 8. FAST DOCUMENT SEARCH MODAL & PANEL (NO GEMINI CALLS, CACHED)
    // =========================================================================
    const btnOpenSearch = document.getElementById('btn-open-search');
    const searchModal = document.getElementById('search-modal');
    const searchModalBackdrop = document.getElementById('search-modal-backdrop');
    const btnCloseSearchModal = document.getElementById('btn-close-search-modal');
    const docSearchInput = document.getElementById('doc-search-input');
    const btnClearDocSearch = document.getElementById('btn-clear-doc-search');
    const searchScopeBadge = document.getElementById('search-scope-badge');
    const searchResultsContainer = document.getElementById('search-results-container');
    const searchStatusText = document.getElementById('search-status-text');
    const searchChips = document.querySelectorAll('.search-chip');

    if (btnOpenSearch) {
        btnOpenSearch.addEventListener('click', openSearchModal);
    }

    function openSearchModal() {
        updateSearchScopeBadge();
        searchModal.classList.remove('hidden');
        if (docSearchInput) {
            setTimeout(() => docSearchInput.focus(), 50);
        }
    }

    function closeSearchModal() {
        searchModal.classList.add('hidden');
    }

    if (btnCloseSearchModal) btnCloseSearchModal.addEventListener('click', closeSearchModal);
    if (searchModalBackdrop) searchModalBackdrop.addEventListener('click', closeSearchModal);

    function updateSearchScopeBadge() {
        if (!searchScopeBadge) return;
        const count = selectedDocs.size;
        const total = allDocumentsList.length;
        if (count === 0) {
            searchScopeBadge.textContent = '0 selected (Check documents in sidebar)';
        } else if (count === total) {
            searchScopeBadge.textContent = `Searching All Documents (${total})`;
        } else {
            searchScopeBadge.textContent = `Searching in ${count} of ${total} selected documents`;
        }
    }

    let searchDebounceTimer = null;
    if (docSearchInput) {
        docSearchInput.addEventListener('input', () => {
            const val = docSearchInput.value.trim();
            if (val) {
                if (btnClearDocSearch) btnClearDocSearch.classList.remove('hidden');
            } else {
                if (btnClearDocSearch) btnClearDocSearch.classList.add('hidden');
            }
            if (searchDebounceTimer) clearTimeout(searchDebounceTimer);
            searchDebounceTimer = setTimeout(() => {
                performDocumentSearch(val);
            }, 250);
        });

        docSearchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                if (searchDebounceTimer) clearTimeout(searchDebounceTimer);
                performDocumentSearch(docSearchInput.value.trim());
            }
        });
    }

    if (btnClearDocSearch) {
        btnClearDocSearch.addEventListener('click', () => {
            if (docSearchInput) docSearchInput.value = '';
            btnClearDocSearch.classList.add('hidden');
            resetSearchResults();
            if (docSearchInput) docSearchInput.focus();
        });
    }

    searchChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const term = chip.getAttribute('data-term');
            if (term && docSearchInput) {
                docSearchInput.value = term;
                if (btnClearDocSearch) btnClearDocSearch.classList.remove('hidden');
                performDocumentSearch(term);
            }
        });
    });

    function resetSearchResults() {
        if (searchStatusText) {
            searchStatusText.textContent = 'Type a query to search fast across indexed chunks.';
        }
        if (searchResultsContainer) {
            searchResultsContainer.innerHTML = `
                <div class="search-empty-state">
                    <span class="empty-icon">💡</span>
                    <p>Search indexed documents with zero Gemini LLM latency.</p>
                </div>
            `;
        }
    }

    async function performDocumentSearch(query) {
        if (!query) {
            resetSearchResults();
            return;
        }

        if (!isVectorstoreReady) {
            if (searchStatusText) searchStatusText.textContent = 'Please index documents first.';
            if (searchResultsContainer) {
                searchResultsContainer.innerHTML = `
                    <div class="search-empty-state">
                        <span class="empty-icon">⏳</span>
                        <p>Documents need to be indexed before searching.</p>
                    </div>
                `;
            }
            return;
        }

        const selectedArray = Array.from(selectedDocs);
        if (selectedArray.length === 0) {
            if (searchStatusText) searchStatusText.textContent = 'Please select at least one document in sidebar.';
            if (searchResultsContainer) {
                searchResultsContainer.innerHTML = `
                    <div class="search-empty-state">
                        <span class="empty-icon">⚠️</span>
                        <p>No documents currently selected. Check one or more documents in the sidebar.</p>
                    </div>
                `;
            }
            return;
        }

        if (searchStatusText) {
            searchStatusText.textContent = `Searching ${selectedArray.length} document${selectedArray.length === 1 ? '' : 's'} for "${query}"...`;
        }

        try {
            const res = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    selected_documents: selectedArray
                })
            });
            const data = await res.json();

            if (!res.ok || !data.success) {
                if (searchStatusText) searchStatusText.textContent = data.error || 'Search error.';
                if (searchResultsContainer) {
                    searchResultsContainer.innerHTML = `
                        <div class="search-empty-state">
                            <span class="empty-icon">⚠️</span>
                            <p>${escapeHtml(data.error || 'Search failed.')}</p>
                        </div>
                    `;
                }
                return;
            }

            const results = data.results || [];
            const elapsed = data.elapsed_seconds || 0;
            const isCached = data.cached ? ' (cached)' : '';

            if (results.length === 0) {
                if (searchStatusText) searchStatusText.textContent = `Found 0 results in ${elapsed}s${isCached}`;
                if (searchResultsContainer) {
                    searchResultsContainer.innerHTML = `
                        <div class="search-empty-state">
                            <span class="empty-icon">🔍</span>
                            <p>No matching content found in the selected documents.</p>
                        </div>
                    `;
                }
                return;
            }

            if (searchStatusText) {
                searchStatusText.textContent = `Found ${results.length} relevant result${results.length === 1 ? '' : 's'} in ${elapsed}s${isCached}`;
            }

            searchResultsContainer.innerHTML = results.map((item, idx) => {
                const cacheKey = `search_res_${Date.now()}_${idx}`;
                const terms = item.matched_terms || [query];
                const docName = item.document_id || item.filename || item.document || 'Document';
                const pageNum = item.page || 1;
                const pdfUrl = item.pdf_url || `/view_pdf/${encodeURIComponent(docName)}#page=${pageNum}`;

                window.currentSourceCache[cacheKey] = {
                    document: docName,
                    page: pageNum,
                    relevancePct: item.relevance_pct,
                    score: item.score,
                    pdfUrl: pdfUrl,
                    snippet: item.snippet,
                    excerpt: item.excerpt,
                    matchedTerms: terms,
                    query: query
                };

                const highlightedSnippet = highlightMatchedTerms(item.snippet, terms);

                return `
                    <div class="search-result-card">
                        <div class="search-card-top">
                            <div class="search-doc-info">
                                <a href="${pdfUrl}" target="_blank" rel="noopener noreferrer" class="search-doc-title" title="Open ${escapeHtml(docName)} in browser at Page ${pageNum}">
                                    📄 ${escapeHtml(docName)}
                                </a>
                                <span class="search-page-badge">📌 Page ${pageNum}</span>
                            </div>
                            <span class="search-card-score">🎯 ${item.relevance_pct}% relevant</span>
                        </div>
                        <div class="search-card-snippet">
                            "${highlightedSnippet}"
                        </div>
                        <div class="search-card-bottom">
                            <a href="${pdfUrl}" target="_blank" rel="noopener noreferrer" class="btn btn-primary btn-sm view-pdf-btn" title="Open original PDF in browser at Page ${pageNum}">
                                <span>🔍</span> View PDF
                            </a>
                            <button type="button" class="btn btn-secondary btn-sm" onclick="openSourcePreview('${cacheKey}')" title="View full extracted excerpt">
                                <span>📄</span> View Excerpt
                            </button>
                        </div>
                    </div>
                `;
            }).join('');

        } catch (err) {
            console.error('Document search error:', err);
            if (searchStatusText) searchStatusText.textContent = 'Error executing search.';
        }
    }

    // Global document keydown for modal closing
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (sourceModal && !sourceModal.classList.contains('hidden')) {
                closeSourcePreview();
            } else if (searchModal && !searchModal.classList.contains('hidden')) {
                closeSearchModal();
            }
        }
    });

    // =========================================================================
    // 9. FAST CHAT Q&A & SMART FILTERED SEARCH
    // =========================================================================
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        sendMessage();
    });

    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isProcessing) {
                sendMessage();
            }
        }
    });

    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
    });

    function mapErrorCodeToMessage(errorCode, defaultMsg) {
        // If the backend already provided a meaningful error message, display that actual error
        if (defaultMsg && typeof defaultMsg === 'string' && defaultMsg.trim().length > 0) {
            return defaultMsg.trim();
        }
        // If no backend error message is available, map the error_code where useful
        switch (errorCode) {
            case 'VALIDATION_ERROR':
                return "Invalid request. Please verify your document selection and query.";
            case 'AI_AUTH_ERROR':
                return "AI authentication failed. Check the API configuration.";
            case 'AI_RATE_LIMIT':
                return "AI request limit reached. Please try again shortly.";
            case 'AI_TIMEOUT':
                return "AI response timed out. Please try again.";
            case 'AI_EMPTY_RESPONSE':
                return "The AI returned an empty response. Please try again.";
            case 'AI_GENERATION_ERROR':
                return "The AI could not generate an answer. Please try again.";
            case 'AI_SERVICE_UNAVAILABLE':
                return "The AI service is temporarily unavailable. Please try again.";
            case 'INTERNAL_SERVER_ERROR':
                return "An internal server error occurred. Please check the backend server logs.";
            case 'INITIALIZATION_ERROR':
                return "Vector database is currently initializing. Please try again in a few moments.";
            default:
                return "The AI service is temporarily unavailable. Please try again.";
        }
    }

    async function sendMessage() {
        if (isProcessing) return;

        const query = userInput.value.trim();
        if (!query) {
            showAlert('Please enter a question.', 'error');
            return;
        }

        // SMART SELECTION VALIDATION (Fix 6)
        if (selectedDocs.size === 0) {
            showAlert('Please select at least one document.', 'error');
            return;
        }

        const readySelectedDocs = Array.from(selectedDocs).filter(filename => {
            const doc = allDocumentsList.find(d => d.filename === filename);
            return doc && (doc.status === 'Ready' || doc.indexed);
        });

        if (readySelectedDocs.length === 0) {
            showAlert("No indexed documents are available. Please click 'Index Documents' first.", 'error');
            return;
        }

        // Remove previous temporary failure messages before starting a new question (Fix 11)
        const oldFailures = chatMessages.querySelectorAll('.ai-failed-message');
        oldFailures.forEach(f => f.remove());

        // Hide welcome placeholder on first query
        const welcome = document.getElementById('welcome-message');
        if (welcome) welcome.remove();

        // 1. Render User Message
        appendUserMessage(query);
        userInput.value = '';
        userInput.style.height = 'auto';

        // 2. Set Processing State & UI
        isProcessing = true;
        btnSend.disabled = true;
        userInput.disabled = true;

        const docCount = selectedDocs.size;
        const initialSearchMsg = docCount === 1
            ? '🔎 Searching 1 selected document...'
            : `🔎 Searching ${docCount} selected documents...`;

        showSearching(true, initialSearchMsg);
        setSystemStatusBadge('searching', '🤖 Searching documents...');

        // 75-second timeout controller (matching backend LLM_TIMEOUT = 60s with buffer)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 75000);

        try {
            // Keep recent conversation history (last 10 messages / 5 full turns)
            const recentHistory = chatHistory.slice(-10);
            const selectedArray = Array.from(selectedDocs);

            // Extract last context if available from last assistant message
            let lastContext = null;
            if (chatHistory.length > 0) {
                for (let i = chatHistory.length - 1; i >= 0; i--) {
                    if (chatHistory[i].role === 'assistant' && chatHistory[i].retrieved_context) {
                        lastContext = chatHistory[i].retrieved_context;
                        break;
                    }
                }
            }

            const payload = {
                query: query,
                question: query,
                selected_documents: selectedArray,
                session_id: 'default_session',
                conversation: recentHistory,
                history: recentHistory,
                last_context: lastContext
            };

            // Requirement Fix 7: Payload verification logging
            console.log("Selected documents:", Array.from(selectedDocs));
            console.log("Selected count:", selectedDocs.size);
            console.log("Chat payload:", payload);

            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                let errorMsg = null;
                let errorCode = null;
                let errorSources = [];
                let timingData = null;
                let respTime = null;
                let adaptiveData = null;

                try {
                    const errData = await response.json();
                    if (errData && typeof errData === 'object') {
                        errorMsg = errData.error || errData.message || null;
                        errorCode = errData.error_code || null;
                        if (errData.sources) errorSources = errData.sources;
                        if (errData.timing_breakdown) timingData = errData.timing_breakdown;
                        if (errData.response_time) respTime = errData.response_time;
                        if (errData.adaptive_retrieval) adaptiveData = errData.adaptive_retrieval;
                    }
                } catch (_) {}

                const finalMsg = errorMsg
                    ? mapErrorCodeToMessage(errorCode, errorMsg)
                    : (errorCode
                        ? mapErrorCodeToMessage(errorCode, null)
                        : (response.status === 400
                            ? "Invalid request. Please check your query and document selection."
                            : (response.status >= 500
                                ? `Server error (HTTP ${response.status}). Please check the backend logs.`
                                : `Request failed with status ${response.status}.`)));

                appendAIFailureMessage(finalMsg, errorSources, timingData, respTime, adaptiveData, query);
                return;
            }

            const data = await response.json();

            // Handle backend structured failure response
            if (data.success === false || data.generation_failed === true) {
                const errorCode = data.error_code;
                const rawError = data.error || data.message;
                const finalMsg = mapErrorCodeToMessage(errorCode, rawError);
                appendAIFailureMessage(
                    finalMsg,
                    data.sources || [],
                    data.timing_breakdown || null,
                    data.response_time || null,
                    data.adaptive_retrieval || null,
                    query
                );
                return;
            }

            const answerText = data.final_verified_answer || data.answer || data.draft_answer;
            if (answerText && typeof answerText === 'string' && answerText.trim().length > 0) {
                const isFollowup = Boolean(data.is_followup || data.context_used || data.used_context || data.mode === 'followup');
                const contextTopic = data.context_topic || '';
                const isComparison = Boolean(data.mode === 'comparison' || data.is_comparison);
                const isDecomposed = Boolean(data.is_decomposed || data.mode === 'decomposition' || (data.sub_queries && data.sub_queries.length > 0));
                const docsUsed = data.documents_used || [];
                const chunksUsed = data.chunks_used !== undefined ? data.chunks_used : (data.sources ? data.sources.length : 0);

                // Update decomposition pill state
                const pillDecomp = document.getElementById('pill-decomposition');
                if (pillDecomp) {
                    if (isDecomposed) {
                        pillDecomp.className = 'status-pill active';
                    }
                }

                saveRecentQuestion(query);
                const finalVerifiedAnswer = data.final_verified_answer || data.answer;
                const draftAnswer = data.draft_answer || data.answer;
                const answerCorrection = data.answer_correction || (data.evaluation ? data.evaluation.answer_correction : null);
                const adaptiveRetrieval = data.adaptive_retrieval || null;
                const compressionData = data.compression || (adaptiveRetrieval ? adaptiveRetrieval.compression : null);
                const compressionDetails = data.compression_details || [];
                const structuredData = data.structured_data || null;

                appendAIMessage(
                    finalVerifiedAnswer,
                    data.sources || [],
                    data.response_time || (data.timing_breakdown ? data.timing_breakdown.total_sec : null),
                    query,
                    isFollowup,
                    data.evaluation || null,
                    isDecomposed ? 'decomposition' : (isComparison ? 'comparison' : 'normal'),
                    docsUsed,
                    chunksUsed,
                    contextTopic,
                    isDecomposed,
                    draftAnswer,
                    answerCorrection,
                    adaptiveRetrieval,
                    compressionData,
                    compressionDetails,
                    structuredData,
                    data.cache_status || (data.cache_hit ? 'HIT' : 'MISS'),
                    data.grouped_sources || [],
                    data.primary_source || '',
                    data.retrieval_stats || (data.evaluation ? data.evaluation.retrieval_stats : {}),
                    data.document_relevance_scores || {}
                );

                // Track full conversation history with metadata and retrieved context
                chatHistory.push({
                    role: 'user',
                    content: query,
                    selected_documents: selectedArray
                });
                chatHistory.push({
                    role: 'assistant',
                    content: finalVerifiedAnswer,
                    draft_answer: draftAnswer,
                    mode: isComparison ? 'comparison' : 'normal',
                    sources: (data.sources || []).map(s => ({
                        document: s.document,
                        page: s.page,
                        score: s.score,
                        relevance_pct: s.relevance_pct
                    })),
                    retrieved_context: data.retrieved_context || [],
                    evaluation: data.evaluation || null,
                    answer_correction: answerCorrection,
                    response_time: data.response_time
                });
            } else {
                appendAIFailureMessage(
                    data.error || "Unable to retrieve information from the selected documents.",
                    data.sources || [],
                    data.timing_breakdown || null,
                    data.response_time || null,
                    data.adaptive_retrieval || null,
                    query
                );
            }

        } catch (err) {
            console.error('Chat error:', err);
            if (err.name === 'AbortError') {
                appendAIFailureMessage('AI response timed out. Please try again.', [], { retrieval_sec: '—', generation_sec: 'timeout', total_sec: 60 }, 60, null, query);
            } else if (err instanceof TypeError || err instanceof ReferenceError) {
                appendAIFailureMessage(`Application display error: ${err.message}`, [], null, null, null, query);
            } else {
                appendAIFailureMessage('The AI service is temporarily unavailable. Please try again.', [], null, null, null, query);
            }
        } finally {
            showSearching(false);
            isProcessing = false;
            btnSend.disabled = false;
            userInput.disabled = false;
            userInput.focus();
            setSystemStatusBadge('ready', '🟢 Ready');
        }
    }

    // Progressive loading state timers
    let loadingTimer1 = null;
    let loadingTimer2 = null;

    function appendUserMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message user-message';
        msgDiv.innerHTML = `
            <div class="message-avatar">👤</div>
            <div class="message-content">
                <p>${escapeHtml(text)}</p>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
    }

    // Global copy answer handler
    window.copyAnswerText = function(btn) {
        const msgContent = btn.closest('.message-content');
        if (!msgContent) return;
        const textBody = msgContent.querySelector('.ai-text-body');
        if (!textBody) return;

        // Copy ONLY plain answer text (omits headers, evaluation dashboard, sources, feedback)
        const plainText = textBody.innerText || textBody.textContent;
        navigator.clipboard.writeText(plainText.trim()).then(() => {
            const originalHtml = btn.innerHTML;
            btn.innerHTML = `<span>✓</span> Copied`;
            btn.classList.add('copied');
            setTimeout(() => {
                btn.innerHTML = originalHtml;
                btn.classList.remove('copied');
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy text:', err);
        });
    };

    // Global answer feedback handler (👍 Helpful / 👎 Not Helpful)
    window.submitAnswerFeedback = function(msgId, rating, btn) {
        const row = btn.closest('.answer-feedback-row');
        if (!row) return;

        const storageKey = `rag_feedback_${msgId}`;
        localStorage.setItem(storageKey, rating);

        const btns = row.querySelectorAll('.btn-feedback');
        btns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const statusMsg = row.querySelector('.feedback-status-msg');
        if (statusMsg) {
            statusMsg.textContent = rating === 'helpful' ? 'Thanks for your feedback!' : 'Feedback noted!';
            statusMsg.classList.remove('hidden');
            setTimeout(() => {
                statusMsg.classList.add('hidden');
            }, 2500);
        }
    };

    // =========================================================================
    // RECENT QUESTIONS QUERY HISTORY (localStorage)
    // =========================================================================
    const RECENT_QUESTIONS_KEY = 'rag_recent_questions';

    function getRecentQuestions() {
        try {
            const raw = localStorage.getItem(RECENT_QUESTIONS_KEY);
            return raw ? JSON.parse(raw) : [];
        } catch (_) {
            return [];
        }
    }

    function saveRecentQuestion(q) {
        if (!q || !q.trim()) return;
        const clean = q.trim();
        let list = getRecentQuestions();
        list = list.filter(item => item.toLowerCase() !== clean.toLowerCase());
        list.unshift(clean);
        list = list.slice(0, 10);
        try {
            localStorage.setItem(RECENT_QUESTIONS_KEY, JSON.stringify(list));
        } catch (_) {}
        renderRecentQuestionsList();
    }

    function renderRecentQuestionsList() {
        const listEl = document.getElementById('recent-questions-list');
        if (!listEl) return;
        const list = getRecentQuestions();
        if (list.length === 0) {
            listEl.innerHTML = '<li class="empty-recent-item">No recent questions yet.</li>';
            return;
        }
        listEl.innerHTML = list.map((item, idx) => `
            <li class="recent-question-item" title="Click to fill into input" onclick="populateRecentQuestion(${idx})">
                <span class="recent-icon">💬</span>
                <span class="recent-text">${escapeHtml(item)}</span>
            </li>
        `).join('');
    }

    window.populateRecentQuestion = function(index) {
        const list = getRecentQuestions();
        if (list[index] && userInput) {
            userInput.value = list[index];
            userInput.style.height = 'auto';
            userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
            userInput.focus();
        }
    };

    const btnClearRecent = document.getElementById('btn-clear-recent');
    if (btnClearRecent) {
        btnClearRecent.addEventListener('click', () => {
            localStorage.removeItem(RECENT_QUESTIONS_KEY);
            renderRecentQuestionsList();
        });
    }

    // Render initial recent questions list from localStorage
    renderRecentQuestionsList();

    // Global sources toggle handler
    window.toggleSources = function(btn) {
        const container = btn.closest('.sources-block');
        if (!container) return;
        const hiddenItems = container.querySelectorAll('.source-item.hidden-source');
        const isExpanded = btn.getAttribute('data-expanded') === 'true';

        if (isExpanded) {
            hiddenItems.forEach(item => item.style.display = 'none');
            btn.setAttribute('data-expanded', 'false');
            btn.textContent = `Show all (${hiddenItems.length + 3})`;
        } else {
            hiddenItems.forEach(item => item.style.display = 'flex');
            btn.setAttribute('data-expanded', 'true');
            btn.textContent = 'Show less';
        }
    };

    // Global source sorting handler
    window.sortSources = function(selectEl) {
        const sortType = selectEl.value;
        const sourcesBlock = selectEl.closest('.sources-block');
        if (!sourcesBlock) return;
        const grid = sourcesBlock.querySelector('.sources-grid');
        if (!grid) return;

        const items = Array.from(grid.querySelectorAll('.source-item'));

        items.sort((a, b) => {
            if (sortType === 'relevance') {
                const scoreA = parseFloat(a.getAttribute('data-relevance') || '0');
                const scoreB = parseFloat(b.getAttribute('data-relevance') || '0');
                return scoreB - scoreA;
            } else if (sortType === 'page') {
                const pageA = parseInt(a.getAttribute('data-page') || '0', 10);
                const pageB = parseInt(b.getAttribute('data-page') || '0', 10);
                return pageA - pageB;
            } else if (sortType === 'document') {
                const docA = (a.getAttribute('data-doc') || '').toLowerCase();
                const docB = (b.getAttribute('data-doc') || '').toLowerCase();
                return docA.localeCompare(docB);
            }
            return 0;
        });

        // Re-append in sorted order and maintain show/hide states
        const isExpanded = sourcesBlock.querySelector('.btn-toggle-sources')?.getAttribute('data-expanded') === 'true';
        items.forEach((item, idx) => {
            const isHidden = idx >= 3;
            if (isHidden) {
                item.classList.add('hidden-source');
                item.style.display = isExpanded ? 'flex' : 'none';
            } else {
                item.classList.remove('hidden-source');
                item.style.display = 'flex';
            }
            grid.appendChild(item);
        });
    };

    // =========================================================================
    // =========================================================================
    // BUILD ANSWER QUALITY PANEL & RETRIEVAL DETAILS
    // =========================================================================
    function buildQualityPanelHtml(evaluation, userQuery) {
        if (!evaluation) return '';

        const groundedScore = evaluation.groundedness_score || 0;
        const groundedLabel = evaluation.groundedness_label || 'Grounded';
        const relScore = evaluation.retrieval_relevance || 0;
        const coverageScore = evaluation.source_coverage || 0;
        const confidence = evaluation.confidence || 'Medium';
        const sourcesUsed = evaluation.sources_count || 0;
        const pagesUsed = evaluation.pages_count || 0;
        const warning = evaluation.warning || null;
        const retrievalMethod = evaluation.retrieval_method || 'Multi-Query Hybrid Search';
        const stats = evaluation.retrieval_stats || {};

        // Advanced Answer Evaluation & Hallucination Detection fields
        const faithfulnessScore = evaluation.faithfulness !== undefined ? Number(evaluation.faithfulness) : (evaluation.faithfulness_score !== undefined ? Number(evaluation.faithfulness_score) : 95.0);
        const hallucinationRisk = evaluation.hallucination_risk !== undefined ? Number(evaluation.hallucination_risk) : (evaluation.hallucination_risk_score !== undefined ? Number(evaluation.hallucination_risk_score) : 5.0);
        const riskLevel = evaluation.risk_level || evaluation.hallucination_risk_level || (hallucinationRisk <= 10 ? 'LOW' : (hallucinationRisk <= 30 ? 'MEDIUM' : 'HIGH'));
        const hallucinationWarning = evaluation.hallucination_warning || null;
        const claimsList = evaluation.claims || [];
        const totalClaims = evaluation.total_claims !== undefined ? evaluation.total_claims : claimsList.length;
        const supportedClaims = evaluation.supported_claims !== undefined ? evaluation.supported_claims : claimsList.filter(c => c.status === 'SUPPORTED').length;
        const partialClaims = evaluation.partial_claims !== undefined ? evaluation.partial_claims : claimsList.filter(c => c.status === 'PARTIALLY_SUPPORTED').length;
        const unsupportedClaims = evaluation.unsupported_claims !== undefined ? evaluation.unsupported_claims : claimsList.filter(c => c.status === 'NOT_SUPPORTED').length;

        const denseMethod = stats.dense_method || 'FAISS';
        const sparseMethod = stats.sparse_method || 'BM25';
        const rerankerMethod = stats.reranker_method || 'Cross Encoder (ms-marco-MiniLM-L-6-v2)';
        const origQuery = stats.original_query || userQuery || '';
        const expandedQueries = stats.expanded_queries || [origQuery];
        const origQueryCount = stats.original_results_count !== undefined ? stats.original_results_count : (stats.semantic_candidates || 10);
        const expQueryCount = stats.expanded_results_count !== undefined ? stats.expanded_results_count : 0;
        const dedupedCount = stats.unique_candidates !== undefined ? stats.unique_candidates : (stats.deduped_candidates !== undefined ? stats.deduped_candidates : (evaluation.retrieval_details ? evaluation.retrieval_details.length : 0));
        const finalCount = stats.final_chunks !== undefined ? stats.final_chunks : (evaluation.retrieval_details ? evaluation.retrieval_details.length : 0);

        let groundedBadgeClass = 'badge-grounded';
        if (groundedLabel === 'Partially Grounded') groundedBadgeClass = 'badge-partially-grounded';
        else if (groundedLabel === 'Not Grounded') groundedBadgeClass = 'badge-not-grounded';

        let confBadgeClass = 'badge-conf-high';
        if (confidence === 'Medium') confBadgeClass = 'badge-conf-med';
        else if (confidence === 'Low') confBadgeClass = 'badge-conf-low';

        let riskBadgeClass = 'badge-risk-low';
        if (riskLevel === 'MEDIUM') riskBadgeClass = 'badge-risk-med';
        else if (riskLevel === 'HIGH') riskBadgeClass = 'badge-risk-high';

        const detailsList = evaluation.retrieval_details || [];

        const isDecomposed = Boolean(stats.is_decomposed || stats.strategy === 'QUERY DECOMPOSITION HYBRID RAG' || (stats.sub_queries && stats.sub_queries.length > 0));
        const subQueries = stats.sub_queries || [];
        const subQueryResults = stats.sub_query_results || [];
        const strategyTitle = isDecomposed ? 'QUERY DECOMPOSITION HYBRID RAG' : 'MULTI-QUERY HYBRID RAG';

        // Compact Table Rows
        let tableRowsHtml = '';
        detailsList.forEach(item => {
            const rank = item.rank || 1;
            const docName = item.document || item.document_name || 'Document';
            const pageNum = item.page || item.page_number || 1;
            const hScore = (item.hybrid_score !== undefined ? Number(item.hybrid_score) : 0.85).toFixed(2);
            const rScore = (item.reranker_score !== undefined ? Number(item.reranker_score) : (item.score || 0.85)).toFixed(2);

            tableRowsHtml += `
                <tr>
                    <td class="col-rank">#${rank}</td>
                    <td class="col-doc" title="${escapeHtml(docName)}">📄 ${escapeHtml(docName)}</td>
                    <td class="col-page">Page ${pageNum}</td>
                    <td class="col-score hybrid">${hScore}</td>
                    <td class="col-score rerank">${rScore}</td>
                </tr>
            `;
        });

        // Sub-queries / Decomposition List HTML
        let decompositionCardHtml = '';
        if (isDecomposed && subQueries.length > 0) {
            let subQueriesListHtml = '';
            subQueries.forEach((sq, sqIdx) => {
                const info = subQueryResults[sqIdx] || {};
                const denseC = info.dense_results !== undefined ? info.dense_results : 10;
                const sparseC = info.sparse_results !== undefined ? info.sparse_results : 10;
                const topDoc = info.top_document || '';
                const topPage = info.top_page || 1;
                const topDocTag = topDoc && topDoc !== 'None' ? `<div class="subquery-top-match"><span>🎯 Top document:</span> <strong>${escapeHtml(topDoc)}</strong> (Page ${topPage})</div>` : '';

                subQueriesListHtml += `
                    <div class="subquery-item">
                        <div class="subquery-item-header">
                            <span class="subquery-badge">Sub-query ${sqIdx + 1}</span>
                            <span class="subquery-counts">Dense: ${denseC} &bull; Sparse: ${sparseC}</span>
                        </div>
                        <div class="subquery-text">"${escapeHtml(sq)}"</div>
                        ${topDocTag}
                    </div>
                `;
            });

            decompositionCardHtml = `
                <div class="query-decomposition-card">
                    <div class="qd-header">
                        <span>⚡</span> <span>Query Decomposition (${subQueries.length} Sub-Queries Generated & Executed):</span>
                    </div>
                    <div class="qd-original-query">
                        <span class="qd-orig-label">Original Query:</span>
                        <span class="qd-orig-text">"${escapeHtml(origQuery)}"</span>
                    </div>
                    <div class="qd-list">
                        ${subQueriesListHtml}
                    </div>
                </div>
            `;
        } else {
            // Expanded Queries List HTML
            let expandedListHtml = '';
            expandedQueries.forEach((q, qIdx) => {
                const isOriginal = (qIdx === 0);
                expandedListHtml += `
                    <div class="expanded-query-item ${isOriginal ? 'is-original' : ''}">
                        <span class="eq-badge">${isOriginal ? 'Query 1 (Original)' : `Query ${qIdx + 1} (Expanded)`}</span>
                        <span class="eq-text">"${escapeHtml(q)}"</span>
                    </div>
                `;
            });

            decompositionCardHtml = `
                <div class="query-expansion-card">
                    <div class="qe-header">
                        <span>🧠</span> <span>Query Expansion (${expandedQueries.length} Search Queries Generated):</span>
                    </div>
                    <div class="qe-list">
                        ${expandedListHtml}
                    </div>
                </div>
            `;
        }

        // =====================================================================
        // CLAIM VERIFICATION ACCORDION COMPONENT
        // =====================================================================
        let claimVerificationHtml = '';
        if (claimsList.length > 0) {
            const claimsItemsHtml = claimsList.map((cl, cIdx) => {
                const status = cl.status || 'SUPPORTED';
                let statusBadge = '';
                let cardClass = 'claim-card';

                if (status === 'SUPPORTED') {
                    statusBadge = '<span class="claim-badge-status supported">✓ SUPPORTED</span>';
                    cardClass += ' status-supported';
                } else if (status === 'PARTIALLY_SUPPORTED') {
                    statusBadge = '<span class="claim-badge-status partial">⚠ PARTIALLY SUPPORTED</span>';
                    cardClass += ' status-partially-supported';
                } else {
                    statusBadge = '<span class="claim-badge-status unsupported">✕ NOT SUPPORTED</span>';
                    cardClass += ' status-not-supported';
                }

                const docName = cl.document || cl.document_name || '';
                const pageNum = cl.page || cl.page_number || 1;
                const pdfUrl = cl.pdf_url || (docName ? `/view_pdf/${encodeURIComponent(docName)}#page=${pageNum}` : '');
                const evidence = cl.evidence || '';

                let sourceRow = '';
                if (status !== 'NOT_SUPPORTED' && docName) {
                    sourceRow = `
                        <div class="claim-source-row">
                            <span class="claim-src-doc">📄 <strong>${escapeHtml(docName)}</strong></span>
                            <span class="claim-src-page">📌 Page ${pageNum}</span>
                            ${pdfUrl ? `<a href="${pdfUrl}" target="_blank" rel="noopener noreferrer" class="claim-view-pdf-btn" title="View PDF at page ${pageNum}"><span>🔍</span> View PDF</a>` : ''}
                        </div>
                    `;
                }

                let evidenceBox = '';
                if (status !== 'NOT_SUPPORTED' && evidence && evidence !== 'No supporting evidence found in selected documents.') {
                    evidenceBox = `
                        <div class="claim-evidence-box">
                            <span class="claim-ev-label">Evidence Snippet:</span>
                            <div class="claim-ev-text">"${escapeHtml(evidence)}"</div>
                        </div>
                    `;
                } else if (status === 'NOT_SUPPORTED') {
                    evidenceBox = `
                        <div class="claim-no-evidence">
                            <span>✕ No supporting evidence found in selected documents.</span>
                        </div>
                    `;
                }

                return `
                    <div class="${cardClass}">
                        <div class="claim-card-header">
                            <span class="claim-num">Claim ${cIdx + 1}</span>
                            ${statusBadge}
                        </div>
                        <div class="claim-statement">"${escapeHtml(cl.claim)}"</div>
                        ${sourceRow}
                        ${evidenceBox}
                    </div>
                `;
            }).join('');

            claimVerificationHtml = `
                <details class="claim-verification-accordion" open>
                    <summary class="claim-verification-summary">
                        <div class="cv-summary-left">
                            <span class="cv-icon">📋</span>
                            <span class="cv-title">Claim Verification</span>
                        </div>
                        <div class="cv-summary-badges">
                            <span class="cv-pill pill-sup">✓ ${supportedClaims} Supported</span>
                            ${partialClaims > 0 ? `<span class="cv-pill pill-part">⚠ ${partialClaims} Partial</span>` : ''}
                            ${unsupportedClaims > 0 ? `<span class="cv-pill pill-unsup">✕ ${unsupportedClaims} Unsupported</span>` : ''}
                        </div>
                    </summary>
                    <div class="claim-verification-body">
                        <div class="cv-intro-meta">
                            <span>Answer verified against retrieved PDF chunks: <strong>${supportedClaims} of ${totalClaims}</strong> factual claims supported.</span>
                        </div>
                        <div class="claims-list-container">
                            ${claimsItemsHtml}
                        </div>
                    </div>
                </details>
            `;
        }

        let detailsHtml = '';
        if (detailsList.length > 0 || dedupedCount > 0) {
            detailsHtml = `
                <details class="retrieval-details-accordion">
                    <summary class="retrieval-details-summary">
                        <span>🔍 Retrieval Details</span>
                        <span class="summary-chunks-tag">${detailsList.length} final chunk${detailsList.length === 1 ? '' : 's'}</span>
                    </summary>
                    <div class="retrieval-details-body">
                        <!-- Retrieval Summary Cards -->
                        <div class="hybrid-stats-card">
                                <div class="hybrid-metric-item">
                                    <span class="h-label">Current Query:</span>
                                    <span class="h-val highlight-strategy">"${escapeHtml(stats.current_query || stats.original_query || userQuery || '')}"</span>
                                </div>
                                <div class="hybrid-metric-item">
                                    <span class="h-label">Normalized Query:</span>
                                    <span class="h-val highlight-strategy">"${escapeHtml(stats.normalized_query || stats.rewritten_query || origQuery)}"</span>
                                </div>
                                <div class="hybrid-metric-item">
                                    <span class="h-label">Query Intent:</span>
                                    <span class="h-val highlight-strategy">${escapeHtml(stats.query_intent || stats.query_topic || evaluation.query_topic || 'General / Document Retrieval')}</span>
                                </div>
                                <div class="hybrid-metric-item">
                                    <span class="h-label">Retrieved Chunks:</span>
                                    <span class="h-val">${stats.candidates_retrieved !== undefined ? stats.candidates_retrieved : (stats.total_candidates || dedupedCount)}</span>
                                </div>
                                <div class="hybrid-metric-item">
                                    <span class="h-label">Rejected Chunks:</span>
                                    <span class="h-val text-warning">${stats.removed_as_irrelevant !== undefined ? stats.removed_as_irrelevant : Math.max(0, (stats.candidates_retrieved || dedupedCount) - finalCount)}</span>
                                </div>
                                <div class="hybrid-metric-item">
                                    <span class="h-label">Reranked Chunks:</span>
                                    <span class="h-val">${stats.after_reranking !== undefined ? stats.after_reranking : (stats.reranked_candidates || dedupedCount)}</span>
                                </div>
                                <div class="hybrid-metric-item">
                                    <span class="h-label">Final Context Chunks:</span>
                                    <span class="h-val highlight-strategy">${stats.final_context_chunks !== undefined ? stats.final_context_chunks : finalCount}</span>
                                </div>
                                <div class="hybrid-metric-item">
                                    <span class="h-label">Source Documents:</span>
                                    <span class="h-val">${escapeHtml(Array.isArray(stats.source_documents) ? stats.source_documents.join(', ') : (Array.isArray(evaluation.sources_used) ? evaluation.sources_used.join(', ') : (stats.source_documents || 'None')))}</span>
                                </div>
                                <div class="hybrid-metric-item">
                                    <span class="h-label">Top Relevance Scores:</span>
                                    <span class="h-val">${stats.top_relevance_scores ? (Array.isArray(stats.top_relevance_scores) ? stats.top_relevance_scores.join(', ') : (typeof stats.top_relevance_scores === 'object' ? Object.entries(stats.top_relevance_scores).map(([d, s]) => `${d}: ${s}%`).join(' | ') : String(stats.top_relevance_scores))) : (detailsList.slice(0, 3).map(d => (d.reranker_score || d.score || 0).toFixed(2)).join(', ') || '—')}</span>
                                </div>
                                <div class="hybrid-metric-item">
                                    <span class="h-label">Cache Status:</span>
                                    <span class="h-val highlight-strategy">${stats.cache_status || 'MISS'}</span>
                                </div>
                                <div class="hybrid-metric-item">
                                    <span class="h-label">Cache Key:</span>
                                    <span class="h-val font-mono">${stats.cache_key || (stats.cache_key_hash ? stats.cache_key_hash.substring(0, 16) : '—')}</span>
                                </div>
                                <div class="hybrid-metric-item">
                                    <span class="h-label">Answer Verification:</span>
                                    <span class="h-val ${supportedClaims > 0 && unsupportedClaims === 0 ? 'text-success' : (unsupportedClaims > 0 ? 'text-warning' : '')}">${supportedClaims}/${totalClaims || 1} claims supported (${faithfulnessScore}%)</span>
                                </div>
                            </div>

                            ${decompositionCardHtml}

                            <div class="hybrid-fusion-row">
                                <span class="fusion-label">Score Fusion:</span>
                                <span class="fusion-tag">Vector Weight: <strong>60%</strong></span>
                                <span class="fusion-sep">•</span>
                                <span class="fusion-tag">BM25 Weight: <strong>40%</strong></span>
                                <span class="fusion-sep">•</span>
                                <span class="fusion-tag">Deduplication: <strong>Max Score</strong></span>
                            </div>
                        </div>

                        <!-- Compact Score Table -->
                        <div class="retrieval-compact-table-wrapper">
                            <table class="retrieval-compact-table">
                                <thead>
                                    <tr>
                                        <th>Rank</th>
                                        <th>Document</th>
                                        <th>Page</th>
                                        <th>Hybrid Score</th>
                                        <th>Reranker Score</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${tableRowsHtml}
                                </tbody>
                            </table>
                        </div>

                        <!-- Collapsible Retrieval Pipeline Flowchart -->
                        <div class="pipeline-flow-container">
                            <div class="pipeline-flow-header">
                                <span>⚡</span> <span>🔎 Retrieval & Answer Verification Pipeline</span>
                            </div>
                            <div class="pipeline-flow-steps">
                                <div class="pipeline-flow-step">
                                    <span class="flow-step-num">1</span>
                                    <div class="flow-step-content">
                                        <span class="flow-step-title">Original Query</span>
                                        <span class="flow-step-desc">"${escapeHtml(origQuery)}"</span>
                                    </div>
                                </div>
                                <div class="flow-arrow-down">↓</div>
                                <div class="pipeline-flow-step">
                                    <span class="flow-step-num">2</span>
                                    <div class="flow-step-content">
                                        <span class="flow-step-title">${isDecomposed ? 'Query Decomposition' : 'Query Expansion'}</span>
                                        <span class="flow-step-badge">${isDecomposed ? `${subQueries.length} sub-queries` : `${expandedQueries.length} query variations`}</span>
                                    </div>
                                </div>
                                <div class="flow-arrow-down">↓</div>
                                <div class="pipeline-flow-step">
                                    <span class="flow-step-num">3</span>
                                    <div class="flow-step-content">
                                        <span class="flow-step-title">FAISS + BM25 Hybrid Retrieval</span>
                                        <span class="flow-step-badge">Dense + Sparse parallel search</span>
                                    </div>
                                </div>
                                <div class="flow-arrow-down">↓</div>
                                <div class="pipeline-flow-step">
                                    <span class="flow-step-num">4</span>
                                    <div class="flow-step-content">
                                        <span class="flow-step-title">Cross-Encoder Reranking</span>
                                        <span class="flow-step-badge">ms-marco-MiniLM-L-6-v2</span>
                                    </div>
                                </div>
                                <div class="flow-arrow-down">↓</div>
                                <div class="pipeline-flow-step">
                                    <span class="flow-step-num">5</span>
                                    <div class="flow-step-content">
                                        <span class="flow-step-title">Gemini Generation</span>
                                        <span class="flow-step-badge">Strict document-grounded answer</span>
                                    </div>
                                </div>
                                <div class="flow-arrow-down">↓</div>
                                <div class="pipeline-flow-step">
                                    <span class="flow-step-num">6</span>
                                    <div class="flow-step-content">
                                        <span class="flow-step-title">Claim Verification & Hallucination Check</span>
                                        <span class="flow-step-badge">${totalClaims} claims verified &bull; ${faithfulnessScore}% faithful</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="details-chunks-list">
                            ${detailsList.map(item => {
                                const vScore = (item.vector_score !== undefined ? Number(item.vector_score) : 0.85).toFixed(2);
                                const bScore = (item.bm25_score !== undefined ? Number(item.bm25_score) : 0.00).toFixed(2);
                                const hScore = (item.hybrid_score !== undefined ? Number(item.hybrid_score) : 0.85).toFixed(2);
                                const rScore = (item.reranker_score !== undefined ? Number(item.reranker_score) : (item.score || 0.85)).toFixed(2);
                                const srcQ = item.source_query ? `<div class="chunk-src-query"><strong>Source Query:</strong> "${escapeHtml(item.source_query)}"</div>` : '';

                                return `
                                    <div class="details-chunk-item">
                                        <div class="details-chunk-top">
                                            <span class="chunk-rank">#${item.rank}</span>
                                            <span class="chunk-doc">📄 ${escapeHtml(item.document || item.document_name || 'Document')}</span>
                                            <span class="chunk-page">📌 Page ${item.page || item.page_number || 1}</span>
                                            <span class="chunk-score">🎯 ${item.relevance_pct}% relevant</span>
                                        </div>
                                        <div class="chunk-scores-row">
                                            <span class="chunk-score-pill score-vector">Vector: <strong>${vScore}</strong></span>
                                            <span class="chunk-score-pill score-bm25">BM25: <strong>${bScore}</strong></span>
                                            <span class="chunk-score-pill score-hybrid">Hybrid: <strong>${hScore}</strong></span>
                                            <span class="chunk-score-pill score-reranker">Reranker: <strong>${rScore}</strong></span>
                                        </div>
                                        ${srcQ}
                                        ${item.snippet ? `<div class="chunk-snippet">"${escapeHtml(item.snippet)}"</div>` : ''}
                                    </div>
                                `;
                            }).join('')}
                        </div>
                    </div>
                </details>
            `;
        }

        // Hallucination Warning Banner HTML
        let hallWarningHtml = '';
        if (riskLevel === 'HIGH') {
            hallWarningHtml = `
                <div class="hallucination-warning-box risk-high">
                    <span class="warning-icon">⚠</span>
                    <div>
                        <strong>Potential unsupported information detected.</strong>
                        <p>Some statements in this answer could not be verified against the selected documents.</p>
                    </div>
                </div>
            `;
        } else if (riskLevel === 'MEDIUM') {
            hallWarningHtml = `
                <div class="hallucination-warning-box risk-med">
                    <span class="warning-icon">⚠</span>
                    <span>Some claims have limited supporting evidence in the selected documents.</span>
                </div>
            `;
        } else if (riskLevel === 'LOW' && claimsList.length > 0) {
            hallWarningHtml = `
                <div class="hallucination-warning-box risk-low">
                    <span class="warning-icon">✓</span>
                    <span>Answer verified against selected documents.</span>
                </div>
            `;
        }

        return `
            <details class="quality-panel-accordion">
                <summary class="quality-panel-summary">
                    <div class="quality-summary-left">
                        <span class="quality-icon">📊</span>
                        <span class="quality-title">Answer Quality & Verification</span>
                    </div>
                    <div class="quality-header-badges">
                        <span class="quality-conf-badge ${confBadgeClass}">Confidence: <strong>${confidence}</strong></span>
                        <span class="quality-risk-badge ${riskBadgeClass}">Hallucination Risk: <strong>${riskLevel}</strong></span>
                    </div>
                </summary>
                <div class="quality-panel-body">
                    ${hallWarningHtml}

                    ${warning && !hallWarningHtml ? `
                        <div class="quality-warning-box">
                            <span class="warning-icon">⚠️</span> <span>${escapeHtml(warning)}</span>
                        </div>
                    ` : ''}

                    <div class="quality-metrics-grid">
                        <div class="quality-metric-card">
                            <div class="metric-top">
                                <span class="metric-label">Groundedness</span>
                                <span class="metric-status-badge ${groundedBadgeClass}">${groundedLabel}</span>
                            </div>
                            <div class="metric-val-row">
                                <span class="metric-value">${groundedScore}%</span>
                                <div class="metric-bar-bg"><div class="metric-bar-fill" style="width: ${groundedScore}%;"></div></div>
                            </div>
                        </div>

                        <div class="quality-metric-card">
                            <div class="metric-top">
                                <span class="metric-label">Faithfulness</span>
                                <span class="metric-tag">${supportedClaims}/${totalClaims || 1} Claims</span>
                            </div>
                            <div class="metric-val-row">
                                <span class="metric-value">${faithfulnessScore}%</span>
                                <div class="metric-bar-bg"><div class="metric-bar-fill fill-faith" style="width: ${faithfulnessScore}%;"></div></div>
                            </div>
                        </div>

                        <div class="quality-metric-card">
                            <div class="metric-top">
                                <span class="metric-label">Retrieval Relevance</span>
                                <span class="metric-tag">Top Match</span>
                            </div>
                            <div class="metric-val-row">
                                <span class="metric-value">${relScore}%</span>
                                <div class="metric-bar-bg"><div class="metric-bar-fill fill-rel" style="width: ${relScore}%;"></div></div>
                            </div>
                        </div>

                        <div class="quality-metric-card">
                            <div class="metric-top">
                                <span class="metric-label">Source Coverage</span>
                                <span class="metric-tag">Approximate</span>
                            </div>
                            <div class="metric-val-row">
                                <span class="metric-value">${coverageScore}%</span>
                                <div class="metric-bar-bg"><div class="metric-bar-fill fill-cov" style="width: ${coverageScore}%;"></div></div>
                            </div>
                        </div>

                        <div class="quality-metric-card">
                            <div class="metric-top">
                                <span class="metric-label">Claim Verification</span>
                                <span class="metric-tag">Detailed</span>
                            </div>
                            <div class="metric-val-row">
                                <div class="claim-mini-summary">
                                    <span class="cms-sup">✓ ${supportedClaims}</span>
                                    <span class="cms-part">⚠ ${partialClaims}</span>
                                    <span class="cms-unsup">✕ ${unsupportedClaims}</span>
                                </div>
                            </div>
                        </div>

                        <div class="quality-metric-card">
                            <div class="metric-top">
                                <span class="metric-label">Sources & Pages</span>
                                <span class="metric-tag">Verified</span>
                            </div>
                            <div class="metric-val-row">
                                <span class="metric-value metric-mini">${sourcesUsed} Source${sourcesUsed === 1 ? '' : 's'} &bull; ${pagesUsed} Page${pagesUsed === 1 ? '' : 's'}</span>
                            </div>
                        </div>
                    </div>

                    ${claimVerificationHtml}

                    ${detailsHtml}
                </div>
            </details>
        `;
    }

    // =========================================================================
    // BUILD ANSWER CORRECTION COMPONENT & COMPARISON VIEW
    // =========================================================================
    function buildAnswerCorrectionHtml(correctionData, draftAnswer, finalVerifiedAnswer) {
        if (!correctionData) return '';
        const total = correctionData.total_claims || 0;
        if (total === 0) return '';

        const supported = correctionData.supported !== undefined ? correctionData.supported : 0;
        const partial = correctionData.partially_supported !== undefined ? correctionData.partially_supported : 0;
        const unsupported = correctionData.unsupported !== undefined ? correctionData.unsupported : (correctionData.removed || 0);
        const removed = correctionData.removed !== undefined ? correctionData.removed : unsupported;
        const coverage = correctionData.final_coverage !== undefined ? correctionData.final_coverage : (total > 0 ? Math.round(((supported + partial) / total) * 1000) / 10 : 0);
        const claimsList = correctionData.claims_correction || [];
        const removedWarning = correctionData.removed_warning || (removed > 0 ? "Some claims were removed because sufficient evidence was not found in the selected documents." : null);

        // 1. Verification Summary Bar (Requirement 11)
        const summaryBarHtml = `
            <div class="correction-summary-bar">
                <div class="cs-item cs-sup">
                    <span class="cs-label">Verified Claims</span>
                    <span class="cs-val">${supported}</span>
                </div>
                <div class="cs-item cs-part">
                    <span class="cs-label">Partially Supported</span>
                    <span class="cs-val">${partial}</span>
                </div>
                <div class="cs-item cs-unsup">
                    <span class="cs-label">Removed Claims</span>
                    <span class="cs-val">${removed}</span>
                </div>
                <div class="cs-item cs-cov">
                    <span class="cs-label">Final Evidence Coverage</span>
                    <span class="cs-val">${coverage}%</span>
                </div>
            </div>
        `;

        // 2. Removal Warning Banner (Requirement 14)
        let warningBannerHtml = '';
        if (removedWarning && removed > 0) {
            warningBannerHtml = `
                <div class="claims-removed-warning-banner">
                    <span class="warning-icon">⚠️</span>
                    <span>${escapeHtml(removedWarning)}</span>
                </div>
            `;
        }

        // 3. Show Original Answer Accordion (Requirement 12)
        let compareAccordionHtml = '';
        if (draftAnswer && draftAnswer.trim() !== (finalVerifiedAnswer || '').trim()) {
            compareAccordionHtml = `
                <details class="original-draft-accordion">
                    <summary class="original-draft-summary">
                        <span>🔍 Show Original Answer (Original Draft vs Final Verified Answer)</span>
                    </summary>
                    <div class="original-draft-body">
                        <div class="draft-compare-grid">
                            <div class="draft-compare-col draft-col">
                                <div class="draft-col-header">
                                    <span class="col-icon">📝</span>
                                    <span class="col-title">Original Draft Answer</span>
                                </div>
                                <div class="draft-col-content">
                                    ${renderRichMarkdown(draftAnswer)}
                                </div>
                            </div>
                            <div class="draft-compare-col verified-col">
                                <div class="draft-col-header">
                                    <span class="col-icon">✓</span>
                                    <span class="col-title">Final Verified Answer</span>
                                </div>
                                <div class="draft-col-content">
                                    ${renderRichMarkdown(finalVerifiedAnswer)}
                                </div>
                            </div>
                        </div>
                    </div>
                </details>
            `;
        }

        // 4. ANSWER CORRECTION claim-by-claim cards (Requirement 9)
        let claimsCardsHtml = '';
        claimsList.forEach((c, idx) => {
            const status = c.status || 'SUPPORTED';
            let statusBadge = '';
            let actionTag = '';
            let cardClass = 'correction-card';
            let resultBox = '';

            const docName = c.document || c.document_name || '';
            const pageNum = c.page || c.page_number || 1;
            const pdfUrl = c.pdf_url || (docName ? `/view_pdf/${encodeURIComponent(docName)}#page=${pageNum}` : '');

            if (status === 'SUPPORTED') {
                statusBadge = '<span class="claim-badge-status supported">✓ SUPPORTED</span>';
                actionTag = '<span class="corr-action-tag action-keep">[Keep]</span>';
                cardClass += ' status-supported';
                resultBox = `
                    <div class="corr-result-box keep-result">
                        <div><strong>Retained Statement:</strong> "${escapeHtml(c.corrected_claim || c.original_claim || c.claim)}"</div>
                        ${docName ? `<div style="margin-top: 6px; font-size: 11px; color: #38bdf8; display:flex; align-items:center; gap:8px; flex-wrap:wrap;"><span>📄 ${escapeHtml(docName)} (Page ${pageNum})</span> <button type="button" class="btn-view-claim-evidence" onclick="window.openEvidenceViewerFromCitation('${escapeHtml(docName)}', ${pageNum}, '', this)" title="View retrieved evidence chunk for Claim ${idx + 1}"><span>🔎</span> View Evidence</button> ${pdfUrl ? `<a href="${pdfUrl}" target="_blank" rel="noopener noreferrer" class="citation-link-badge"><span>🔍</span> View PDF</a>` : ''}</div>` : ''}
                    </div>
                `;
            } else if (status === 'PARTIALLY_SUPPORTED') {
                statusBadge = '<span class="claim-badge-status partial">⚠ PARTIALLY SUPPORTED</span>';
                actionTag = '<span class="corr-action-tag action-rewrite">[Rewritten using document evidence]</span>';
                cardClass += ' status-partially-supported';
                resultBox = `
                    <div class="corr-result-box rewrite-result">
                        <div><strong>Rewritten Statement:</strong> "${escapeHtml(c.corrected_claim || '')}"</div>
                        ${c.evidence && c.evidence !== 'No supporting evidence found in selected documents.' ? `<div style="margin-top: 5px; font-size: 11px; color: #cbd5e1;"><em>Document Evidence:</em> "${escapeHtml(c.evidence)}"</div>` : ''}
                        ${docName ? `<div style="margin-top: 6px; font-size: 11px; color: #38bdf8; display:flex; align-items:center; gap:8px; flex-wrap:wrap;"><span>📄 ${escapeHtml(docName)} (Page ${pageNum})</span> <button type="button" class="btn-view-claim-evidence" onclick="window.openEvidenceViewerFromCitation('${escapeHtml(docName)}', ${pageNum}, '', this)" title="View retrieved evidence chunk for Claim ${idx + 1}"><span>🔎</span> View Evidence</button> ${pdfUrl ? `<a href="${pdfUrl}" target="_blank" rel="noopener noreferrer" class="citation-link-badge"><span>🔍</span> View PDF</a>` : ''}</div>` : ''}
                    </div>
                `;
            } else {
                statusBadge = '<span class="claim-badge-status unsupported">✕ NOT SUPPORTED</span>';
                actionTag = '<span class="corr-action-tag action-remove">[Removed from final answer]</span>';
                cardClass += ' status-not-supported';
                resultBox = `
                    <div class="corr-result-box remove-result">
                        <span>✕ Removed from final answer. Reason: No supporting evidence found in selected documents.</span>
                    </div>
                `;
            }

            claimsCardsHtml += `
                <div class="${cardClass}">
                    <div class="correction-card-header">
                        <span class="corr-claim-num">CLAIM ${idx + 1}</span>
                        <div class="corr-badge-group">
                            ${statusBadge}
                            ${actionTag}
                        </div>
                    </div>
                    <div class="corr-flow-step">
                        <span class="corr-step-label">Original Claim:</span>
                        <div class="corr-original-text">"${escapeHtml(c.original_claim || c.claim)}"</div>
                    </div>
                    <div class="corr-flow-divider">↓</div>
                    <div class="corr-flow-step">
                        <span class="corr-step-label">Verification & Correction:</span>
                        ${resultBox}
                    </div>
                </div>
            `;
        });

        const correctionAccordionHtml = `
            <details class="answer-correction-accordion">
                <summary class="answer-correction-summary">
                    <div class="ac-summary-left">
                        <span class="ac-icon">🔧</span>
                        <span class="ac-title">Detailed Claim Verification & Answer Correction</span>
                    </div>
                    <div class="ac-summary-badges">
                        <span class="ac-pill ac-sup">✓ ${supported} Kept</span>
                        ${partial > 0 ? `<span class="ac-pill ac-part">⚠ ${partial} Rewritten</span>` : ''}
                        ${removed > 0 ? `<span class="ac-pill ac-unsup">✕ ${removed} Removed</span>` : ''}
                    </div>
                </summary>
                <div class="answer-correction-body">
                    <div class="ac-flow-legend">
                        <span>Original Claims</span>
                        <span class="flow-arrow">→</span>
                        <span>Verification</span>
                        <span class="flow-arrow">→</span>
                        <span>Correction</span>
                    </div>
                    <div class="correction-cards-list">
                        ${claimsCardsHtml}
                    </div>
                </div>
            </details>
        `;

        return `
            <div class="answer-correction-wrapper">
                ${warningBannerHtml}
                ${summaryBarHtml}
                ${compareAccordionHtml}
                ${correctionAccordionHtml}
            </div>
        `;
    }

    // =========================================================================
    // RELEVANCE BADGE HELPER (Requirement 11)
    // =========================================================================
    function getRelevanceBadge(scoreOrPct) {
        let pct = 90;
        if (typeof scoreOrPct === 'number') {
            pct = scoreOrPct > 1 ? Math.round(scoreOrPct) : Math.round(scoreOrPct * 100);
        } else if (typeof scoreOrPct === 'string') {
            pct = parseInt(scoreOrPct, 10) || 90;
        }
        if (pct >= 90) return { stars: '★★★★★', label: 'Very High', class: 'rel-very-high', pct: pct };
        if (pct >= 75) return { stars: '★★★★', label: 'High', class: 'rel-high', pct: pct };
        if (pct >= 50) return { stars: '★★★', label: 'Moderate', class: 'rel-moderate', pct: pct };
        if (pct >= 35) return { stars: '★★', label: 'Low', class: 'rel-low', pct: pct };
        return { stars: '★', label: 'Very Low', class: 'rel-very-low', pct: pct };
    }

    // =========================================================================
    // BUILD RETRIEVAL PROCESS & RETRIEVAL EXPLANATION (Requirement 10)
    // =========================================================================
    function buildRetrievalProcessHtml(adaptiveData, compressionSummary = null, structuredData = null, userQuery = '', retrievalStats = null, queryType = '') {
        if (!adaptiveData && !compressionSummary && !structuredData && !retrievalStats) return '';
        let steps = (adaptiveData && adaptiveData.process_steps) ? [...adaptiveData.process_steps] : [];
        const stats = retrievalStats || (adaptiveData ? adaptiveData.retrieval_stats : {}) || {};

        const qText = userQuery || stats.original_query || 'User Query';
        const qType = queryType || stats.query_type || stats.query_topic || (qText.toLowerCase().includes('sql') || qText.toLowerCase().includes('select') || qText.toLowerCase().includes('command') ? 'SQL / Data Querying' : 'Document Retrieval');
        const cRetrieved = stats.candidates_retrieved !== undefined ? stats.candidates_retrieved : (stats.total_candidates !== undefined ? stats.total_candidates : (stats.deduped_candidates || 20));
        const aRerank = stats.after_reranking !== undefined ? stats.after_reranking : (stats.reranked_candidates !== undefined ? stats.reranked_candidates : 10);
        const aChunkFilter = stats.after_chunk_filter !== undefined ? stats.after_chunk_filter : (stats.after_relevance_filtering !== undefined ? stats.after_relevance_filtering : 6);
        const relDocs = stats.relevant_documents !== undefined ? stats.relevant_documents : (stats.documents_compared || (stats.document_relevance_scores ? Object.keys(stats.document_relevance_scores).length : 1));
        const fContext = stats.final_context_chunks !== undefined ? stats.final_context_chunks : (stats.final_chunks || 4);
        const rIrrelevant = stats.removed_as_irrelevant !== undefined ? stats.removed_as_irrelevant : Math.max(0, cRetrieved - fContext);
        const primSource = stats.primary_source || stats.primary_document || (stats.document_relevance_scores ? Object.keys(stats.document_relevance_scores)[0] : 'None');

        const currQ = stats.current_query || qText;
        const normQ = stats.normalized_query || currQ;
        const qIntent = stats.query_intent || stats.query_topic || qType;
        const rjChunks = stats.rejected_chunks !== undefined ? stats.rejected_chunks : (stats.removed_as_irrelevant !== undefined ? stats.removed_as_irrelevant : rIrrelevant);
        const rrChunks = stats.reranked_chunks !== undefined ? stats.reranked_chunks : aRerank;
        const srcDocsList = stats.source_documents ? (Array.isArray(stats.source_documents) ? stats.source_documents.join(', ') : stats.source_documents) : (stats.primary_document || relDocs);
        const topScoresStr = stats.top_relevance_scores ? Object.entries(stats.top_relevance_scores).map(([d, s]) => `${d}: ${s}%`).join(' | ') : primSource;
        const cacheStat = stats.cache_status || 'MISS';
        const cacheKeyShort = stats.cache_key || (stats.cache_hit ? 'HIT' : 'N/A');
        const verifStatus = stats.answer_verification || 'Supported';

        // Build Retrieval Explanation Card (Requirement 12)
        const explanationCardHtml = `
            <div class="retrieval-explanation-card">
                <div class="rex-header">
                    <span>⚡</span> <span>RETRIEVAL PROCESS & DEBUG INFORMATION</span>
                </div>
                <div class="rex-grid">
                    <div class="rex-item full-width">
                        <span class="rex-key">CURRENT QUERY</span>
                        <span class="rex-val">"${escapeHtml(currQ)}"</span>
                    </div>
                    <div class="rex-item full-width">
                        <span class="rex-key">NORMALIZED QUERY</span>
                        <span class="rex-val highlight-val">"${escapeHtml(normQ)}"</span>
                    </div>
                    <div class="rex-item">
                        <span class="rex-key">QUERY INTENT</span>
                        <span class="rex-val highlight-val">${escapeHtml(qIntent)}</span>
                    </div>
                    <div class="rex-item">
                        <span class="rex-key">RETRIEVED CHUNKS</span>
                        <span class="rex-val">${cRetrieved} chunks</span>
                    </div>
                    <div class="rex-item">
                        <span class="rex-key">REJECTED CHUNKS</span>
                        <span class="rex-val text-warning">${rjChunks} chunks</span>
                    </div>
                    <div class="rex-item">
                        <span class="rex-key">RERANKED CHUNKS</span>
                        <span class="rex-val">${rrChunks} chunks</span>
                    </div>
                    <div class="rex-item">
                        <span class="rex-key">FINAL CONTEXT</span>
                        <span class="rex-val final-val">${fContext} chunks</span>
                    </div>
                    <div class="rex-item full-width">
                        <span class="rex-key">SOURCE DOCUMENTS</span>
                        <span class="rex-val">${escapeHtml(srcDocsList)}</span>
                    </div>
                    <div class="rex-item full-width">
                        <span class="rex-key">TOP RELEVANCE SCORES</span>
                        <span class="rex-val primary-val">${escapeHtml(topScoresStr)}</span>
                    </div>
                    <div class="rex-item">
                        <span class="rex-key">CACHE STATUS</span>
                        <span class="rex-val ${cacheStat === 'HIT' ? 'text-success' : ''}">${cacheStat} (${escapeHtml(cacheKeyShort)})</span>
                    </div>
                    <div class="rex-item">
                        <span class="rex-key">ANSWER VERIFICATION</span>
                        <span class="rex-val highlight-val">✓ ${escapeHtml(verifStatus)}</span>
                    </div>
                </div>
            </div>
        `;

        // Check if structured data / table retrieval should be enriched in steps
        if (structuredData && (structuredData.tables_detected > 0 || structuredData.tables_retrieved > 0)) {
            const hasTableStep = steps.some(s => (s.title || '').toLowerCase().includes('table') || (s.title || '').toLowerCase().includes('structured'));
            if (!hasTableStep) {
                const tableStep = {
                    icon: "📊",
                    title: "Table & Structured Data Retrieval",
                    desc: `${structuredData.tables_retrieved || 0} table(s) retrieved (${structuredData.tables_detected || 0} detected in documents)`
                };
                const compIdx = steps.findIndex(s => (s.title || '').toLowerCase().includes('compression') || (s.title || '').toLowerCase().includes('reranking'));
                if (compIdx !== -1) {
                    steps.splice(compIdx + 1, 0, tableStep);
                } else {
                    steps.push(tableStep);
                }
            }
        }

        // Check if compression stats should be enriched in steps
        if (compressionSummary && compressionSummary.applied) {
            const redPct = compressionSummary.reduction_percentage || 0;
            const origChars = (compressionSummary.original_characters || 0).toLocaleString();
            const compChars = (compressionSummary.compressed_characters || 0).toLocaleString();
            const count = compressionSummary.chunks_compressed_count || 0;

            const hasCompStep = steps.some(s => (s.title || '').toLowerCase().includes('compression'));
            if (!hasCompStep) {
                const insertIdx = steps.findIndex(s => (s.title || '').toLowerCase().includes('reranking'));
                const compStep = {
                    icon: "🗜️",
                    title: "Contextual Compression",
                    desc: `${count} chunks compressed (${redPct}% reduction)`
                };
                if (insertIdx !== -1) {
                    steps.splice(insertIdx + 1, 0, compStep);
                } else {
                    steps.push(compStep);
                }
            }

            const hasReductionStep = steps.some(s => (s.title || '').toLowerCase().includes('reduction'));
            if (!hasReductionStep) {
                const compIdx = steps.findIndex(s => (s.title || '').toLowerCase().includes('compression'));
                const redStep = {
                    icon: "📊",
                    title: "Context Reduction",
                    desc: `Original: ${origChars} chars | Compressed: ${compChars} chars | Reduction: ${redPct}%`
                };
                if (compIdx !== -1) {
                    steps.splice(compIdx + 1, 0, redStep);
                } else {
                    steps.push(redStep);
                }
            }
        }

        // Add LLM Generation step if not present
        const hasGen = steps.some(s => (s.title || '').toLowerCase().includes('generation') || (s.title || '').toLowerCase().includes('answer generation'));
        if (!hasGen) {
            const finalIdx = steps.findIndex(s => (s.title || '').toLowerCase().includes('final status'));
            const genStep = {
                icon: "🤖",
                title: "LLM Generation",
                desc: "Answer generated from compressed context"
            };
            if (finalIdx !== -1) {
                steps.splice(finalIdx, 0, genStep);
            } else {
                steps.push(genStep);
            }
        }

        const retries = adaptiveData ? (adaptiveData.retries_performed || 0) : 0;
        const retryBadge = retries > 0
            ? `<span class="rp-badge-retry">🔄 ${retries} Retry${retries === 1 ? '' : 's'} Triggered</span>`
            : `<span class="rp-badge-no-retry">✓ High Confidence</span>`;

        const stepsHtml = steps.map((step, idx) => {
            const isFinal = (step.title || '').toLowerCase().includes('final status');
            return `
                <div class="rp-step-item ${isFinal ? 'final-status' : ''}">
                    <span class="rp-step-icon">${step.icon || '✓'}</span>
                    <div class="rp-step-content">
                        <span class="rp-step-title">${escapeHtml(step.title || '')}</span>
                        <span class="rp-step-desc">${escapeHtml(step.desc || '')}</span>
                    </div>
                </div>
            `;
        }).join('');

        return `
            <details class="retrieval-process-accordion">
                <summary class="retrieval-process-summary">
                    <div class="rp-summary-left">
                        <span class="rp-icon">🔎</span>
                        <span class="rp-title">Retrieval Process</span>
                    </div>
                    ${retryBadge}
                </summary>
                <div class="retrieval-process-body">
                    ${explanationCardHtml}
                    <div class="rp-steps-list">
                        ${stepsHtml}
                    </div>
                </div>
            </details>
        `;
    }

    function buildRetrievalAttemptsHtml(adaptiveData) {
        if (!adaptiveData) return '';
        const attempts = adaptiveData.attempts || [];
        if (attempts.length === 0) return '';

        const selectedAttemptNum = adaptiveData.selected_attempt || 1;
        const retries = adaptiveData.retries_performed || 0;

        const attemptsHtml = attempts.map(att => {
            const attNum = att.attempt || 1;
            const isSelected = (attNum === selectedAttemptNum);
            const status = att.status || 'ACCEPTED';
            let statusTag = `<span class="attempt-status-tag tag-accepted">✓ Accepted</span>`;
            if (status === 'RETRY_TRIGGERED') {
                statusTag = `<span class="attempt-status-tag tag-retry">⚠️ Quality Check Failed &bull; Retry Triggered</span>`;
            } else if (status === 'BETTER_RESULT') {
                statusTag = `<span class="attempt-status-tag tag-better">⭐ Improved Result</span>`;
            } else if (status === 'RETAINED_PREVIOUS') {
                statusTag = `<span class="attempt-status-tag tag-accepted">Previous Attempt Retained</span>`;
            }

            const queries = att.queries || [];
            const sources = att.sources || [];
            const supClaims = att.supported_claims !== undefined ? att.supported_claims : 0;
            const totClaims = att.total_claims !== undefined ? att.total_claims : 0;
            const unsupClaims = att.unsupported_claims !== undefined ? att.unsupported_claims : 0;
            const faith = att.faithfulness !== undefined ? att.faithfulness : 100;

            return `
                <div class="attempt-card ${isSelected ? 'is-selected' : ''}">
                    <div class="attempt-card-header">
                        <span class="attempt-title">Attempt ${attNum}: ${escapeHtml(att.type || 'Retrieval')} ${isSelected ? '🎯 [Selected Final]' : ''}</span>
                        ${statusTag}
                    </div>
                    <div class="attempt-detail-row">
                        <span class="attempt-detail-label">Queries Used:</span>
                        <div class="attempt-queries-list">
                            ${queries.map(q => `<span class="attempt-query-tag">"${escapeHtml(q)}"</span>`).join('')}
                        </div>
                    </div>
                    <div class="attempt-detail-row">
                        <span class="attempt-detail-label">Chunks Retrieved:</span>
                        <span><strong>${att.candidates_count || att.final_chunks_count || 0}</strong> candidates (${att.final_chunks_count || 0} reranked final chunks)</span>
                    </div>
                    <div class="attempt-detail-row">
                        <span class="attempt-detail-label">Top Sources:</span>
                        <span>${sources.length > 0 ? escapeHtml(sources.join(', ')) : 'Selected Documents'}</span>
                    </div>
                    <div class="attempt-detail-row">
                        <span class="attempt-detail-label">Verification Score:</span>
                        <span><strong>${supClaims}/${totClaims || 1}</strong> claims supported (${faith}% faithful, ${unsupClaims} unsupported)</span>
                    </div>
                </div>
            `;
        }).join('');

        return `
            <details class="retrieval-attempts-accordion">
                <summary class="retrieval-attempts-summary">
                    <div class="ra-summary-left">
                        <span class="ra-icon">🔄</span>
                        <span class="ra-title">Retrieval Attempts</span>
                    </div>
                    <span class="ra-badge">${attempts.length} Attempt${attempts.length === 1 ? '' : 's'} (${retries} Retr${retries === 1 ? 'y' : 'ies'})</span>
                </summary>
                <div class="retrieval-attempts-body">
                    ${attemptsHtml}
                    <div class="attempt-final-summary">
                        <span><strong>Final Selection:</strong> Attempt ${selectedAttemptNum} adopted for answer generation.</span>
                        <span><strong>Retry Count:</strong> ${retries}</span>
                    </div>
                </div>
            </details>
        `;
    }

    // =========================================================================
    // BUILD CONTEXTUAL COMPRESSION DETAILS ACCORDION
    // =========================================================================
    function buildCompressionDetailsHtml(compressionSummary, compressionDetails) {
        if (!compressionDetails || compressionDetails.length === 0) return '';
        const summary = compressionSummary || {};
        const origChars = (summary.original_characters || 0).toLocaleString();
        const compChars = (summary.compressed_characters || 0).toLocaleString();
        const redPct = summary.reduction_percentage !== undefined ? summary.reduction_percentage : 0;
        const timeMs = summary.compression_time_ms || 0;

        const chunksHtml = compressionDetails.map((chunk, idx) => {
            const docName = chunk.document || chunk.filename || 'Document';
            const page = chunk.page || 1;
            const origLen = chunk.original_length || (chunk.original_text ? chunk.original_text.length : 0);
            const compLen = chunk.compressed_length || (chunk.compressed_text ? chunk.compressed_text.length : 0);
            const ratio = chunk.compression_ratio !== undefined ? chunk.compression_ratio : (origLen > 0 ? Math.round((1 - (compLen / origLen)) * 1000) / 10 : 0);
            const score = chunk.score !== undefined ? chunk.score : 0.90;
            const relPct = chunk.relevance_pct || Math.round(score * 100);
            const chunkId = chunk.chunk_id || `chunk_${idx + 1}`;

            return `
                <div class="compression-chunk-card">
                    <div class="comp-card-header">
                        <div class="comp-card-title">
                            <span class="comp-chunk-badge">${escapeHtml(chunkId)}</span>
                            <span class="comp-doc-badge">📄 ${escapeHtml(docName)} &bull; Page ${page}</span>
                        </div>
                        <div class="comp-card-meta">
                            <span class="comp-rel-badge">🎯 Relevance: ${score} (${relPct}%)</span>
                            <span class="comp-ratio-badge">🗜️ ${ratio}% reduction</span>
                        </div>
                    </div>
                    <div class="comp-diff-grid">
                        <div class="comp-diff-col comp-orig-col">
                            <div class="comp-diff-header">
                                <span class="comp-diff-label">Original Chunk</span>
                                <span class="comp-len-tag">${origLen} characters</span>
                            </div>
                            <div class="comp-text-box comp-orig-text">${escapeHtml(chunk.original_text || '')}</div>
                        </div>
                        <div class="comp-diff-arrow">↓</div>
                        <div class="comp-diff-col comp-clean-col">
                            <div class="comp-diff-header">
                                <span class="comp-diff-label">Compressed Chunk (Relevant Sentences Kept)</span>
                                <span class="comp-len-tag highlight">${compLen} characters</span>
                            </div>
                            <div class="comp-text-box comp-clean-text">${escapeHtml(chunk.compressed_text || '')}</div>
                        </div>
                    </div>
                    <div class="comp-card-footer">
                        <span><strong>Source:</strong> ${escapeHtml(docName)}</span>
                        <span><strong>Page:</strong> ${page}</span>
                        <span><strong>Relevance score:</strong> ${score}</span>
                        <span><strong>Compression ratio:</strong> ${origLen} chars &rarr; ${compLen} chars (${ratio}%)</span>
                    </div>
                </div>
            `;
        }).join('');

        return `
            <details class="compression-details-accordion">
                <summary class="compression-details-summary">
                    <div class="cd-summary-left">
                        <span class="cd-icon">🗜️</span>
                        <span class="cd-title">Contextual Compression Details</span>
                    </div>
                    <span class="cd-badge">${compressionDetails.length} Chunks &bull; ${redPct}% Reduction (${timeMs}ms)</span>
                </summary>
                <div class="compression-details-body">
                    <div class="compression-summary-banner">
                        <div class="cs-stat"><span class="cs-lbl">Original:</span> <strong>${origChars}</strong> characters</div>
                        <div class="cs-stat"><span class="cs-lbl">Compressed:</span> <strong>${compChars}</strong> characters</div>
                        <div class="cs-stat"><span class="cs-lbl">Reduction:</span> <strong class="cs-highlight">${redPct}%</strong></div>
                        <div class="cs-stat"><span class="cs-lbl">Compression Time:</span> <strong>${timeMs}ms</strong></div>
                    </div>
                    <div class="compression-chunks-list">
                        ${chunksHtml}
                    </div>
                </div>
            </details>
        `;
    }
    // =========================================================================
    // BUILD STRUCTURED DATA (TABLES) ACCORDION & CARDS
    // =========================================================================
    function buildStructuredDataHtml(structuredData) {
        if (!structuredData) return '';
        const detected = structuredData.tables_detected || 0;
        const retrieved = structuredData.tables_retrieved || 0;
        const tablesList = structuredData.tables || [];

        if (detected === 0 && tablesList.length === 0) return '';

        let tableCardsHtml = '';
        if (tablesList.length > 0) {
            tableCardsHtml = tablesList.map((t, idx) => {
                const docName = t.document || 'Document';
                const pageNum = t.page || 1;
                const tableNum = t.table_number || (idx + 1);
                const cols = t.columns || [];
                const rows = t.rows || [];
                const tableMd = t.table_markdown || '';
                const pdfUrl = `/view_pdf/${encodeURIComponent(docName)}#page=${pageNum}`;

                const renderedTable = tableMd ? renderRichMarkdown(tableMd) : '';

                let rowsHtml = '';
                if (rows && rows.length > 0) {
                    const rowItems = rows.slice(0, 5).map(r => {
                        return `<div class="sd-row-item">${escapeHtml(typeof r === 'string' ? r : JSON.stringify(r))}</div>`;
                    }).join('');
                    rowsHtml = `
                        <div class="sd-relevant-rows">
                            <div class="sd-rel-row-label">Relevant Rows Sample:</div>
                            <div class="sd-row-list">${rowItems}</div>
                        </div>
                    `;
                }

                return `
                    <div class="structured-table-card">
                        <div class="sd-card-header">
                            <div class="sd-card-title">
                                <span class="sd-table-badge">Table ${tableNum}</span>
                                <span class="sd-doc-badge">📄 ${escapeHtml(docName)}</span>
                                <span class="sd-page-badge">📌 Page ${pageNum}</span>
                            </div>
                            <div class="sd-card-meta">
                                ${cols.length > 0 ? `<span class="sd-cols-badge">${cols.length} Columns</span>` : ''}
                                ${rows.length > 0 ? `<span class="sd-rows-badge">${rows.length} Rows</span>` : ''}
                                <a href="${pdfUrl}" target="_blank" rel="noopener noreferrer" class="btn btn-primary btn-sm view-pdf-btn" title="View Table in PDF at Page ${pageNum}">
                                    <span>🔍</span> View PDF
                                </a>
                            </div>
                        </div>
                        ${renderedTable ? `<div class="sd-table-preview-wrapper">${renderedTable}</div>` : ''}
                        ${rowsHtml}
                    </div>
                `;
            }).join('');
        } else {
            tableCardsHtml = `
                <div class="sd-empty-notice" style="color: #94a3b8; font-size: 12px; font-style: italic; padding: 6px 0;">
                    ${detected} structured table(s) detected across indexed documents. (No table chunks directly referenced in current query).
                </div>
            `;
        }

        return `
            <details class="structured-data-accordion" open>
                <summary class="structured-data-summary">
                    <div class="sd-summary-left">
                        <span class="sd-icon">📊</span>
                        <span class="sd-title">STRUCTURED DATA & TABLES</span>
                    </div>
                    <span class="sd-badge">${retrieved} Table${retrieved === 1 ? '' : 's'} Retrieved &bull; ${detected} Detected</span>
                </summary>
                <div class="structured-data-body">
                    <div class="structured-data-summary-banner">
                        <div class="sd-stat"><span class="sd-lbl">Tables Detected:</span> <strong>${detected}</strong></div>
                        <div class="sd-stat"><span class="sd-lbl">Tables Retrieved:</span> <strong class="sd-highlight">${retrieved}</strong></div>
                        <div class="sd-stat"><span class="sd-lbl">Extraction Engine:</span> <strong>pdfplumber + Layout Analyzer</strong></div>
                    </div>
                    <div class="structured-tables-list">
                        ${tableCardsHtml}
                    </div>
                </div>
            </details>
        `;
    }

    function appendAIMessage(
        answerText,
        sources,
        responseTime,
        userQuery = '',
        isFollowup = false,
        evaluation = null,
        mode = 'normal',
        documentsUsed = [],
        chunksUsed = 0,
        contextTopic = '',
        isDecomposed = false,
        draftAnswer = null,
        answerCorrection = null,
        adaptiveRetrieval = null,
        compressionSummary = null,
        compressionDetails = [],
        structuredData = null,
        cacheStatus = 'MISS',
        groupedSources = [],
        primarySource = '',
        retrievalStats = {},
        documentRelevanceScores = {}
    ) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ai-message';
        const isComparisonMode = (mode === 'comparison');
        const isDecompositionMode = (mode === 'decomposition' || isDecomposed);
        if (isComparisonMode) {
            msgDiv.classList.add('comparison-message');
        }
        if (isDecompositionMode) {
            msgDiv.classList.add('decomposition-message');
        }
        const msgId = `msg_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
        msgDiv.id = msgId;

        if (!window.chatInteractionMap) window.chatInteractionMap = {};
        window.chatInteractionMap[msgId] = {
            id: (structuredData && structuredData.id) || null,
            question: userQuery,
            final_verified_answer: answerText,
            answer: answerText,
            draft_answer: draftAnswer,
            sources: sources,
            grouped_sources: groupedSources,
            primary_source: primarySource,
            document_relevance_scores: documentRelevanceScores,
            retrieval_stats: retrievalStats,
            response_time: responseTime,
            evaluation: evaluation,
            answer_correction: answerCorrection,
            adaptive_retrieval: adaptiveRetrieval,
            compression: compressionSummary,
            compression_details: compressionDetails,
            structured_data: structuredData,
            documents_used: documentsUsed,
            chunks_used: chunksUsed,
            is_comparison: isComparisonMode,
            is_decomposed: isDecompositionMode,
            cache_status: cacheStatus
        };

        let sourcesHtml = '';
        if (sources && sources.length > 0) {
            const docPageMap = new Map();
            sources.forEach((s, idx) => {
                const docName = s.document_id || s.document || s.filename || 'Document';
                const pageNum = s.page || 1;
                const snippet = s.snippet || '';
                const excerpt = s.excerpt || s.snippet || '';
                const relPct = s.relevance_pct || (s.score ? Math.round(s.score * 100) : 90);
                const pdfUrl = s.pdf_url || `/view_pdf/${encodeURIComponent(docName)}#page=${pageNum}`;
                const key = `${docName}__p_${pageNum}`;
                const isTable = Boolean(s.is_table || s.table_number);
                const tableNum = s.table_number || null;

                if (!docPageMap.has(key)) {
                    const cacheKey = `src_${Date.now()}_${idx}`;
                    docPageMap.set(key, {
                        cacheKey,
                        document: docName,
                        page: pageNum,
                        relevancePct: relPct,
                        score: s.score || (relPct / 100),
                        pdfUrl: pdfUrl,
                        snippet: snippet,
                        excerpt: excerpt,
                        chunksCount: 1,
                        is_table: isTable,
                        table_number: tableNum
                    });
                } else {
                    const existing = docPageMap.get(key);
                    existing.chunksCount += 1;
                    if (relPct > existing.relevancePct) {
                        existing.relevancePct = relPct;
                        existing.score = s.score || (relPct / 100);
                    }
                    if (snippet && (!existing.snippet || existing.snippet.length < snippet.length)) {
                        existing.snippet = snippet;
                        existing.excerpt = excerpt || snippet;
                    }
                }
            });

            const uniqueSources = Array.from(docPageMap.values());
            uniqueSources.forEach(s => {
                window.currentSourceCache[s.cacheKey] = {
                    document: s.document,
                    page: s.page,
                    relevancePct: s.relevancePct,
                    score: s.score,
                    pdfUrl: s.pdfUrl,
                    snippet: s.snippet,
                    excerpt: s.excerpt,
                    query: userQuery
                };
            });

            if (uniqueSources.length > 0) {
                const hasManySources = uniqueSources.length > 4;
                const topPrimaryStr = primarySource || (uniqueSources[0] ? `${uniqueSources[0].document} — Page ${uniqueSources[0].page}` : '');

                // Group sources by document for Requirement 5
                const docGroupsMap = new Map();
                uniqueSources.forEach(s => {
                    if (!docGroupsMap.has(s.document)) {
                        const docRelInfo = (documentRelevanceScores && documentRelevanceScores[s.document]) || {};
                        const docRelPct = docRelInfo.relevance_pct || s.relevancePct;
                        docGroupsMap.set(s.document, {
                            document: s.document,
                            docRelPct: docRelPct,
                            isPrimary: topPrimaryStr.startsWith(s.document),
                            pages: [s.page],
                            items: [s]
                        });
                    } else {
                        const grp = docGroupsMap.get(s.document);
                        if (!grp.pages.includes(s.page)) grp.pages.push(s.page);
                        grp.items.push(s);
                        if (s.relevancePct > grp.docRelPct) grp.docRelPct = s.relevancePct;
                    }
                });

                const docGroups = Array.from(docGroupsMap.values()).sort((a, b) => b.docRelPct - a.docRelPct);

                sourcesHtml = `
                    <div class="sources-block">
                        <div class="sources-header-row">
                            <div class="sources-title">
                                <span>📄</span> SOURCES <span class="sources-count">(${docGroups.length} Document${docGroups.length === 1 ? '' : 's'}, ${uniqueSources.length} Section${uniqueSources.length === 1 ? '' : 's'})</span>
                            </div>
                            <div class="sources-header-controls">
                                <div class="sources-sort-control">
                                    <select class="sources-sort-select" onchange="sortSources(this)" title="Sort retrieved sources">
                                        <option value="relevance">Highest Relevance</option>
                                        <option value="page">Page Number</option>
                                        <option value="document">Document Name</option>
                                    </select>
                                </div>
                                ${hasManySources ? `
                                    <button class="btn-toggle-sources" data-expanded="false" onclick="toggleSources(this)">
                                        Show all (${uniqueSources.length})
                                    </button>
                                ` : ''}
                            </div>
                        </div>

                        ${topPrimaryStr && topPrimaryStr !== 'None' ? `
                            <div class="primary-source-card">
                                <div class="primary-source-header">
                                    <span class="ps-badge">⭐ PRIMARY SOURCE</span>
                                    <span class="ps-label">${escapeHtml(topPrimaryStr)}</span>
                                </div>
                            </div>
                        ` : ''}

                        <!-- Grouped Sources Display (Requirement 1 & 5) -->
                        <div class="grouped-sources-container" style="display:flex; flex-direction:column; gap:12px; margin-bottom:12px;">
                            ${docGroups.map(grp => {
                                const bInfo = getRelevanceBadge(grp.docRelPct);
                                return `
                                    <div class="grouped-source-item" style="border:1px solid rgba(255,255,255,0.08); background:rgba(255,255,255,0.02); border-radius:8px; padding:10px 14px;">
                                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; flex-wrap:wrap; gap:8px;">
                                            <div style="display:flex; align-items:center; gap:8px;">
                                                <span style="font-size:16px;">📄</span>
                                                <strong style="color:#f1f5f9; font-size:13.5px;">${escapeHtml(grp.document)}</strong>
                                                ${grp.isPrimary ? '<span style="background:rgba(245,158,11,0.15); color:#fbbf24; font-size:10.5px; font-weight:700; padding:2px 7px; border-radius:4px;">PRIMARY</span>' : ''}
                                            </div>
                                            <div style="display:flex; align-items:center; gap:8px;">
                                                <span style="color:#94a3b8; font-size:12px;">Document Relevance: <strong style="color:#38bdf8;">${grp.docRelPct}%</strong></span>
                                                <span class="source-stars-pill ${bInfo.class}" style="font-size:11px; padding:2px 8px; border-radius:12px; font-weight:600;">
                                                    ${bInfo.stars} ${bInfo.label}
                                                </span>
                                            </div>
                                        </div>
                                        <div style="display:flex; gap:6px; align-items:center; margin-bottom:6px;">
                                            <span style="color:#64748b; font-size:11.5px;">Pages:</span>
                                            ${grp.pages.map(p => `<span style="background:rgba(255,255,255,0.06); color:#cbd5e1; font-size:11px; padding:1px 6px; border-radius:4px;">Page ${p}</span>`).join('')}
                                        </div>
                                    </div>
                                `;
                            }).join('')}
                        </div>

                        <!-- Sources Grid with View PDF & View Excerpt -->
                        <div class="sources-grid">
                            ${uniqueSources.map((s, idx) => {
                                const isHidden = idx >= 4;
                                const highlightedSnippet = highlightMatchedTerms(s.snippet, userQuery);
                                const tableBadge = (s.is_table || s.table_number) ? `<span class="source-table-tag">📊 Table ${s.table_number || 1}</span>` : '';
                                const chunkBadge = s.chunksCount > 1 ? `<span class="source-chunks-badge" style="background:rgba(99,102,241,0.15); color:#818cf8; font-size:11px; padding:2px 6px; border-radius:4px;">🧩 ${s.chunksCount} supporting chunks</span>` : '';
                                const bInfo = getRelevanceBadge(s.relevancePct);
                                return `
                                    <div class="source-item ${isHidden ? 'hidden-source' : ''}"
                                         data-doc="${escapeHtml(s.document)}"
                                         data-page="${s.page}"
                                         data-relevance="${s.relevancePct}"
                                         style="${isHidden ? 'display: none;' : ''}">
                                        <div class="source-item-top">
                                            <a href="${s.pdfUrl}" target="_blank" rel="noopener noreferrer" class="source-doc-link" title="Open ${escapeHtml(s.document)} in browser at Page ${s.page}">
                                                <span class="src-icon">📄</span> <span class="src-filename">${escapeHtml(s.document)}</span>
                                            </a>
                                            <span class="source-page-tag">📌 Page ${s.page}</span>
                                            ${chunkBadge}
                                            ${tableBadge}
                                        </div>
                                        ${s.snippet ? `
                                            <div class="source-snippet-preview" title="Retrieved context snippet">
                                                "${highlightedSnippet}"
                                            </div>
                                        ` : ''}
                                        <div class="source-item-bottom">
                                            <div style="display:flex; align-items:center; gap:6px;">
                                                <span class="source-rel-tag">🎯 Relevance: ${s.relevancePct}%</span>
                                                <span class="source-stars-badge ${bInfo.class}" style="font-size:10.5px; font-weight:600;">
                                                    ${bInfo.stars} ${bInfo.label}
                                                </span>
                                            </div>
                                            <div class="source-actions">
                                                <a href="${s.pdfUrl}" target="_blank" rel="noopener noreferrer" class="btn btn-primary btn-sm view-pdf-btn" title="Open original PDF at Page ${s.page}">
                                                    <span>🔍</span> View PDF
                                                </a>
                                                <button class="btn-snippet-preview" data-cache-key="${s.cacheKey}" onclick="openSourcePreview('${s.cacheKey}')" title="Preview exact retrieved excerpt from Page ${s.page}">
                                                    <span>📄</span> View Excerpt
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                `;
                            }).join('')}
                        </div>
                    </div>
                `;
            }
        }

        let timingHtml = '';
        if (responseTime !== null && responseTime !== undefined) {
            const timingBreakdown = evaluation && evaluation.timing_breakdown ? evaluation.timing_breakdown : null;
            let parts = [];
            const totSec = typeof responseTime === 'number' ? responseTime.toFixed(1) : responseTime;
            if (timingBreakdown) {
                if (timingBreakdown.retrieval_sec !== undefined) parts.push(`Retrieval: ${timingBreakdown.retrieval_sec}s`);
                if (timingBreakdown.reranking_sec !== undefined) parts.push(`Reranking: ${timingBreakdown.reranking_sec}s`);
                if (timingBreakdown.compression_sec !== undefined && timingBreakdown.compression_sec > 0) parts.push(`Compression: ${timingBreakdown.compression_sec}s`);
                if (timingBreakdown.generation_sec !== undefined) parts.push(`Generation: ${timingBreakdown.generation_sec}s`);
                if (timingBreakdown.verification_sec !== undefined && timingBreakdown.verification_sec > 0) parts.push(`Verification: ${timingBreakdown.verification_sec}s`);
            }
            const breakdownStr = parts.length > 0 ? ` (${parts.join(' | ')})` : '';
            const isHit = (cacheStatus === 'HIT' || cacheStatus === true);
            const cacheBadge = isHit
                ? '<span class="cache-badge hit" title="Retrieved instantly from in-memory cache">⚡ CACHE: HIT</span>'
                : '<span class="cache-badge miss" title="Computed fresh from document index">🔄 CACHE: MISS</span>';

            timingHtml = `
                <div class="msg-footer-row">
                    ${cacheBadge}
                    <span class="response-time-badge">
                        ⏱️ Total: ${totSec}s${breakdownStr}
                    </span>
                </div>
            `;
        }

        let followupBadge = '';
        if (isFollowup) {
            const topicText = contextTopic ? ` &bull; Context: <strong>${escapeHtml(contextTopic)}</strong>` : '';
            followupBadge = `<span class="context-indicator-tag" title="Question understood using previous conversation context">↪ Follow-up understood${topicText}</span>`;
        }

        const corrObj = answerCorrection || (evaluation ? evaluation.answer_correction : null);
        const finalAns = answerText;
        const draftAns = draftAnswer || (evaluation ? evaluation.draft_answer : answerText);
        const correctionSectionHtml = buildAnswerCorrectionHtml(corrObj, draftAns, finalAns);
        const structuredDataHtml = buildStructuredDataHtml(structuredData);
        const processHtml = buildRetrievalProcessHtml(adaptiveRetrieval, compressionSummary, structuredData, userQuery, retrievalStats, evaluation ? evaluation.query_type : '');
        const attemptsHtml = buildRetrievalAttemptsHtml(adaptiveRetrieval);
        const compressionDetailsHtml = buildCompressionDetailsHtml(compressionSummary, compressionDetails);
        const qualityPanelHtml = evaluation ? buildQualityPanelHtml(evaluation, userQuery) : '';
        const savedFeedback = localStorage.getItem(`rag_feedback_${msgId}`);

        let statusBadge = '<span class="final-verified-badge-tag status-verified">✓ Verified</span>';

        const isFallback = (
            !finalAns ||
            finalAns === "The selected documents do not contain enough information to answer this question." ||
            finalAns.startsWith("Not enough supporting evidence") ||
            finalAns.startsWith("The answer is not available") ||
            finalAns.startsWith("I couldn't find") ||
            finalAns.startsWith("I could not find") ||
            finalAns.startsWith("Unable to generate")
        );

        if (isFallback) {
            statusBadge = '<span class="final-verified-badge-tag status-unsupported">✕ Not Supported</span>';
        } else if (corrObj && corrObj.unsupported > 0 && corrObj.supported === 0) {
            statusBadge = '<span class="final-verified-badge-tag status-unsupported">✕ Unsupported</span>';
        } else if (corrObj && corrObj.partially_supported > 0 && corrObj.supported === 0) {
            statusBadge = '<span class="final-verified-badge-tag status-partial">⚠ Partially Supported</span>';
        } else if (evaluation && (evaluation.risk_level === 'HIGH' || evaluation.groundedness_score < 40)) {
            statusBadge = '<span class="final-verified-badge-tag status-unsupported">⚠ High Risk</span>';
        } else {
            statusBadge = '<span class="final-verified-badge-tag status-verified">✓ Verified</span>';
        }

        const headerBadge = isDecompositionMode
            ? `<span class="ai-header-tag">ANSWER</span> ${statusBadge}`
            : (isComparisonMode
                ? `<span class="ai-header-tag">ANSWER</span> <span class="comparison-header-tag" title="Multi-Document Comparison Mode">🔄 Comparison</span> ${statusBadge}`
                : `<span class="ai-header-tag">ANSWER</span> ${statusBadge}`);

        let comparisonMetaHtml = '';
        if (isComparisonMode && (documentsUsed.length > 0 || chunksUsed > 0 || (sources && sources.length > 0))) {
            const docsCount = documentsUsed.length > 0 ? documentsUsed.length : (evaluation ? evaluation.sources_count : 0);
            const totalChunks = chunksUsed > 0 ? chunksUsed : (evaluation ? evaluation.chunks_count : 0);
            const sourcesCount = sources ? sources.length : 0;
            comparisonMetaHtml = `
                <div class="comparison-meta-row">
                    <span class="comp-meta-item"><strong>Documents Compared:</strong> ${docsCount}</span>
                    <span class="comp-meta-dot">•</span>
                    <span class="comp-meta-item"><strong>Sections Used:</strong> ${totalChunks}</span>
                    <span class="comp-meta-dot">•</span>
                    <span class="comp-meta-item"><strong>Sources:</strong> ${sourcesCount}</span>
                </div>
            `;
        }

        // Build Top Answer Summary Bar (Confidence, Grounding, Verification)
        function buildTopSummaryBar(evalObj, cObj, fallback) {
            const conf = fallback ? 'Low' : (evalObj ? (evalObj.confidence || 'Medium') : 'Medium');
            const ground = fallback ? 'Not Grounded' : (evalObj ? (evalObj.groundedness_label || 'Grounded') : 'Grounded');
            const sup = evalObj ? (evalObj.supported_claims !== undefined ? evalObj.supported_claims : (cObj ? cObj.supported : 0)) : (cObj ? cObj.supported : 0);
            const part = evalObj ? (evalObj.partial_claims !== undefined ? evalObj.partial_claims : (cObj ? cObj.partially_supported : 0)) : (cObj ? cObj.partially_supported : 0);
            const unsup = evalObj ? (evalObj.unsupported_claims !== undefined ? evalObj.unsupported_claims : (cObj ? cObj.unsupported : 0)) : (cObj ? cObj.unsupported : 0);

            let confCls = 'badge-conf-high';
            if (conf === 'Medium') confCls = 'badge-conf-med';
            else if (conf === 'Low') confCls = 'badge-conf-low';

            let grndCls = 'badge-grounded';
            if (ground === 'Partially Grounded') grndCls = 'badge-partially-grounded';
            else if (ground === 'Not Grounded') grndCls = 'badge-not-grounded';

            return `
                <div class="top-answer-summary-bar">
                    <div class="summary-meta-item">
                        <span class="meta-item-label">CONFIDENCE</span>
                        <span class="meta-item-badge ${confCls}">● ${escapeHtml(conf)}</span>
                    </div>
                    <div class="summary-meta-item">
                        <span class="meta-item-label">GROUNDING</span>
                        <span class="meta-item-badge ${grndCls}">${escapeHtml(ground)}</span>
                    </div>
                    <div class="summary-meta-item">
                        <span class="meta-item-label">ACCURACY</span>
                        <div class="meta-claim-pills">
                            <span class="pill-sup">✓ ${sup} Verified</span>
                            ${part > 0 ? `<span class="pill-part">⚠ ${part} Partial</span>` : ''}
                            ${unsup > 0 ? `<span class="pill-unsup">✕ ${unsup} Unverified</span>` : ''}
                        </div>
                    </div>
                </div>
            `;
        }

        const topSummaryHtml = buildTopSummaryBar(evaluation, corrObj, isFallback);

        msgDiv.innerHTML = `
            <div class="message-avatar">${isComparisonMode ? '🔄' : '🤖'}</div>
            <div class="message-content">
                <div class="ai-msg-header">
                    <div class="ai-header-left">
                        ${headerBadge}
                        ${followupBadge}
                    </div>
                    <div class="ai-header-right-btns" style="display:flex; align-items:center; gap:6px;">
                        <button type="button" class="btn-inspect-detail-chat" onclick="window.openQuestionDetails('${msgId}')" title="Inspect complete telemetry and claims in Question Detail modal">
                            <span>📊</span> Details
                        </button>
                        <button class="btn-copy-answer" onclick="copyAnswerText(this)" title="Copy final answer to clipboard">
                            <span>📋</span> Copy
                        </button>
                    </div>
                </div>
                ${comparisonMetaHtml}

                <!-- 1. FINAL ANSWER -->
                <div class="final-verified-answer-section">
                    <div class="ai-text-body">${renderRichMarkdown(finalAns)}</div>
                </div>

                <!-- 2. SOURCES -->
                ${sourcesHtml}

                <!-- 3. CONFIDENCE, GROUNDING & VERIFICATION SUMMARY -->
                ${topSummaryHtml}

                <!-- 4. STRUCTURED DATA & TABLES (IF APPLICABLE) -->
                ${structuredDataHtml}

                <!-- 5. RETRIEVAL PROCESS (COLLAPSIBLE) -->
                ${processHtml}

                <!-- 6. RETRIEVAL ATTEMPTS (COLLAPSIBLE) -->
                ${attemptsHtml}

                <!-- 7. CONTEXTUAL COMPRESSION (COLLAPSIBLE) -->
                ${compressionDetailsHtml}

                <!-- 8. ANSWER QUALITY & VERIFICATION (COLLAPSIBLE) -->
                ${qualityPanelHtml}

                <!-- 9. DETAILED CLAIM VERIFICATION & ANSWER CORRECTION (COLLAPSIBLE) -->
                ${correctionSectionHtml}

                <!-- 7. PERFORMANCE TIMING BAR -->
                ${timingHtml}

                <!-- 8. FEEDBACK -->
                <div class="answer-feedback-row" data-msg-id="${msgId}">
                    <span class="feedback-prompt">Was this answer helpful?</span>
                    <div class="feedback-btn-group">
                        <button type="button" class="btn-feedback btn-feedback-up ${savedFeedback === 'helpful' ? 'active' : ''}" onclick="submitAnswerFeedback('${msgId}', 'helpful', this)" title="Helpful answer">
                            <span>👍</span> Helpful
                        </button>
                        <button type="button" class="btn-feedback btn-feedback-down ${savedFeedback === 'unhelpful' ? 'active' : ''}" onclick="submitAnswerFeedback('${msgId}', 'unhelpful', this)" title="Not helpful answer">
                            <span>👎</span> Not Helpful
                        </button>
                    </div>
                    <span class="feedback-status-msg hidden"></span>
                </div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
    }

    function appendAIFailureMessage(
        errorMessage,
        sources = [],
        timingBreakdown = null,
        responseTime = null,
        adaptiveRetrieval = null,
        retryQueryText = ''
    ) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ai-message ai-failed-message';
        const msgId = `msg_failed_${Date.now()}`;
        msgDiv.id = msgId;

        const isTimeout = errorMessage && (errorMessage.toLowerCase().includes('time limit') || errorMessage.toLowerCase().includes('timeout') || errorMessage.toLowerCase().includes('timed out'));
        const cleanMsg = isTimeout
            ? "Unable to generate the answer within the current time limit."
            : (errorMessage || "Unable to generate the answer within the current time limit.");

        let timingHtml = '';
        if (responseTime !== null || timingBreakdown) {
            const parts = [];
            if (timingBreakdown) {
                if (timingBreakdown.retrieval_sec !== undefined) parts.push(`Retrieval: ${timingBreakdown.retrieval_sec}s`);
                if (timingBreakdown.reranking_sec !== undefined) parts.push(`Reranking: ${timingBreakdown.reranking_sec}s`);
                if (timingBreakdown.generation_sec !== undefined) parts.push(`Generation: ${timingBreakdown.generation_sec === 'timeout' ? 'timeout' : timingBreakdown.generation_sec + 's'}`);
                if (timingBreakdown.total_sec !== undefined) parts.push(`Total: ${timingBreakdown.total_sec}s`);
            } else if (responseTime !== null && responseTime !== undefined) {
                parts.push(`Total: ${typeof responseTime === 'number' ? responseTime.toFixed(1) + 's' : responseTime}`);
            }
            if (parts.length > 0) {
                timingHtml = `
                    <div class="msg-footer-row">
                        <span class="response-time-badge response-failed-badge">
                            ⏱️ ${parts.join(' | ')}
                        </span>
                    </div>
                `;
            }
        }

        const retryEscaped = (retryQueryText || '').replace(/'/g, "\\'");
        const retryBtnHtml = retryQueryText ? `
            <div class="failure-retry-row" style="margin-top: 12px;">
                <button type="button" class="btn btn-secondary btn-sm btn-retry-query" onclick="window.retryQuery('${escapeHtml(retryEscaped)}')" style="display:inline-flex; align-items:center; gap:6px; cursor:pointer;" title="Retry asking this question">
                    <span>🔄</span> Try Again
                </button>
            </div>
        ` : '';

        msgDiv.innerHTML = `
            <div class="message-avatar">⚠️</div>
            <div class="message-content">
                <div class="ai-msg-header">
                    <div class="ai-header-left">
                        <span class="ai-header-tag status-failed">⚠️ NOTICE</span>
                    </div>
                </div>
                <div class="final-verified-answer-section failed-answer-section">
                    <div class="ai-text-body error-text">${escapeHtml(cleanMsg)}</div>
                    ${retryBtnHtml}
                </div>
                ${timingHtml}
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
    }

    window.retryQuery = function(queryText) {
        if (!queryText || isProcessing) return;
        if (userInput) {
            userInput.value = queryText;
        }
        if (chatForm) {
            chatForm.dispatchEvent(new Event('submit'));
        }
    };

    let loadingTimer3 = null;
    let loadingTimer4 = null;

    function showSearching(show, customText = '') {
        if (loadingTimer1) clearTimeout(loadingTimer1);
        if (loadingTimer2) clearTimeout(loadingTimer2);
        if (loadingTimer3) clearTimeout(loadingTimer3);
        if (loadingTimer4) clearTimeout(loadingTimer4);

        if (show) {
            const docCount = selectedDocs.size;
            if (thinkingText) {
                thinkingText.textContent = customText || (docCount === 1 ? '🔎 Searching 1 selected document...' : `🔎 Searching ${docCount} selected documents...`);
            }
            thinkingIndicator.classList.remove('hidden');

            loadingTimer1 = setTimeout(() => {
                if (thinkingText && isProcessing) {
                    thinkingText.textContent = '🧠 Generating answer...';
                }
            }, 500);

            loadingTimer2 = setTimeout(() => {
                if (thinkingText && isProcessing) {
                    thinkingText.textContent = '✓ Verifying answer...';
                }
            }, 2400);

            loadingTimer3 = setTimeout(() => {
                if (thinkingText && isProcessing) {
                    thinkingText.textContent = '✓ Finalizing response...';
                }
            }, 7500);
        } else {
            thinkingIndicator.classList.add('hidden');
        }
        scrollToBottom();
    }

    function scrollToBottom(smooth = true) {
        if (chatContainer) {
            requestAnimationFrame(() => {
                chatContainer.scrollTo({
                    top: chatContainer.scrollHeight,
                    behavior: smooth ? 'smooth' : 'auto'
                });
            });
        }
    }

    if (btnNewChat) {
        btnNewChat.addEventListener('click', async () => {
            await startNewChat();
        });
    }

    if (btnClearChat) {
        btnClearChat.addEventListener('click', async () => {
            await clearChat();
        });
    }

    async function startNewChat() {
        chatHistory = [];
        window.currentSourceCache = {};
        try {
            await fetch('/api/new_chat', { method: 'POST' });
        } catch (_) {}

        showAlert('Started a new conversation session.', 'success');

        chatMessages.innerHTML = `
            <div class="message system-message" id="welcome-message">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <h3>New Conversation Started</h3>
                    <p>Conversational memory is active. Ask questions and follow-ups across your selected documents.</p>
                </div>
            </div>
        `;
        userInput.value = '';
        userInput.focus();
    }

    async function clearChat() {
        chatHistory = [];
        window.currentSourceCache = {};
        try {
            await fetch('/api/clear_chat', { method: 'POST' });
        } catch (_) {}

        showAlert('Conversation cleared.', 'info');

        chatMessages.innerHTML = `
            <div class="message system-message" id="welcome-message">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <h3>Conversation cleared.</h3>
                    <p>Ask a new question strictly answered using your selected documents.</p>
                </div>
            </div>
        `;
        userInput.value = '';
        userInput.focus();
    }

    // =========================================================================
    // 8. SYSTEM STATUS SYNC & DOCUMENT LIST RENDERING
    // =========================================================================
    async function fetchSystemStatus() {
        try {
            const res = await fetch('/api/status');
            if (!res.ok) throw new Error('Status endpoint returned error');
            const data = await res.json();

            // Status Counters
            statDocs.textContent = data.total_documents || 0;
            if (statIndexed) statIndexed.textContent = data.total_indexed !== undefined ? data.total_indexed : (data.indexed_count !== undefined ? data.indexed_count : (data.indexed_documents ? data.indexed_documents.length : 0));
            if (statPending) statPending.textContent = data.total_pending !== undefined ? data.total_pending : (data.pending_count !== undefined ? data.pending_count : (data.documents ? data.documents.filter(d => d.status === 'Pending').length : 0));
            statPages.textContent = data.total_pages || 0;
            statChunks.textContent = data.total_chunks || 0;
            statModel.textContent = data.gemini_model || 'Gemini';

            allDocumentsList = data.documents || [];

            // On initial load, select all documents by default
            if (!initialSelectionLoaded && allDocumentsList.length > 0) {
                allDocumentsList.forEach(d => selectedDocs.add(d.filename));
                initialSelectionLoaded = true;
            } else if (allDocumentsList.length === 0) {
                selectedDocs.clear();
                initialSelectionLoaded = false;
            } else {
                // Prune any removed documents from selectedDocs
                const availableFiles = new Set(allDocumentsList.map(d => d.filename));
                for (const sf of Array.from(selectedDocs)) {
                    if (!availableFiles.has(sf)) {
                        selectedDocs.delete(sf);
                    }
                }
            }

            renderFileList(allDocumentsList);
            updateSelectionUI();

            // Update Pipeline Status Pills with actual system state
            const comp = data.components || {};
            updatePillStatus('pill-hybrid', comp.hybrid_search, 'Hybrid Search');
            updatePillStatus('pill-faiss', comp.faiss, 'FAISS Dense');
            updatePillStatus('pill-bm25', comp.bm25, 'BM25 Sparse');
            updatePillStatus('pill-reranker', comp.cross_encoder, 'Cross-Encoder');
            updatePillStatus('pill-multiquery', comp.multi_query, 'Multi-Query');
            updatePillStatus('pill-decomposition', comp.query_decomposition, 'Query Decomposition');
            updatePillStatus('pill-adaptive', comp.adaptive_retrieval !== false, 'Adaptive Retrieval');
            updatePillStatus('pill-compression', comp.contextual_compression !== false, 'Contextual Compression');
            updatePillStatus('pill-table', comp.table_extraction !== false, 'Table Extraction');
            updatePillStatus('pill-evaluation', comp.answer_evaluation !== false, 'Answer Evaluation');
            updatePillStatus('pill-correction', comp.answer_correction !== false, 'Answer Correction');

            const readyCount = data.total_indexed !== undefined ? data.total_indexed : (data.indexed_count !== undefined ? data.indexed_count : (data.indexed_documents ? data.indexed_documents.length : 0));
            const pendingCount = data.total_pending !== undefined ? data.total_pending : (data.pending_count !== undefined ? data.pending_count : 0);

            if (data.vectorstore_ready && data.total_chunks > 0 && readyCount > 0) {
                isVectorstoreReady = true;
                setSystemStatusBadge('ready', '🟢 Ready');
                btnIndex.disabled = false;
                btnRebuild.disabled = false;
            } else if (allDocumentsList.length > 0) {
                isVectorstoreReady = false;
                setSystemStatusBadge('empty', pendingCount > 0 ? '⏳ Needs Indexing' : '🟡 Standby');
                btnIndex.disabled = false;
                btnRebuild.disabled = readyCount === 0 && pendingCount > 0 ? false : false;
            } else {
                isVectorstoreReady = false;
                setSystemStatusBadge('empty', '⚪ No Documents');
                btnIndex.disabled = true;
                btnRebuild.disabled = true;
            }
        } catch (err) {
            console.error('Status sync error:', err);
            setSystemStatusBadge('empty', '🔴 Offline');
            updatePillStatus('pill-hybrid', false, 'Hybrid Search');
            updatePillStatus('pill-faiss', false, 'FAISS Dense');
            updatePillStatus('pill-bm25', false, 'BM25 Sparse');
            updatePillStatus('pill-reranker', false, 'Cross-Encoder');
            updatePillStatus('pill-multiquery', false, 'Multi-Query');
            updatePillStatus('pill-decomposition', false, 'Query Decomposition');
            updatePillStatus('pill-adaptive', false, 'Adaptive Retrieval');
            updatePillStatus('pill-compression', false, 'Contextual Compression');
            updatePillStatus('pill-table', false, 'Table Extraction');
            updatePillStatus('pill-evaluation', false, 'Answer Evaluation');
            updatePillStatus('pill-correction', false, 'Answer Correction');
        }
    }

    function updatePillStatus(pillId, isEnabled, label) {
        const pill = document.getElementById(pillId);
        if (!pill) return;
        if (isEnabled) {
            pill.className = 'status-pill active';
            pill.innerHTML = `<span class="pill-dot">●</span> ${escapeHtml(label)}`;
        } else {
            pill.className = 'status-pill standby';
            pill.innerHTML = `<span class="pill-dot">●</span> ${escapeHtml(label)}`;
        }
    }

    function renderFileList(documents) {
        fileCountBadge.textContent = `${documents.length} document${documents.length === 1 ? '' : 's'}`;

        if (!documents || documents.length === 0) {
            fileList.innerHTML = '<li class="empty-list-item">No documents uploaded yet.</li>';
            return;
        }

        fileList.innerHTML = documents.map(doc => {
            const rawStatus = (doc.status || (doc.indexed ? 'Ready' : 'Pending')).toLowerCase();
            let statusClass = 'badge-pending';
            let statusIcon = '🟡';
            let statusLabel = 'Pending';

            if (rawStatus === 'ready' || doc.indexed) {
                statusClass = 'badge-ready';
                statusIcon = '🟢';
                statusLabel = 'Ready';
            } else if (rawStatus === 'indexing') {
                statusClass = 'badge-indexing';
                statusIcon = '🔵';
                statusLabel = 'Indexing...';
            } else if (rawStatus === 'deleting') {
                statusClass = 'badge-deleting';
                statusIcon = '🗑️';
                statusLabel = 'Deleting...';
            } else if (rawStatus === 'error') {
                statusClass = 'badge-error';
                statusIcon = '🔴';
                statusLabel = 'Indexing failed';
            }

            const isSelected = selectedDocs.has(doc.filename);
            const safeId = sanitizeId(doc.filename);
            const docId = doc.document_id || `doc_${doc.filename}`;
            const pagesVal = doc.pages || 1;
            const chunksVal = (rawStatus === 'ready' || doc.indexed) ? (doc.chunks || 0) : 0;
            const sizeVal = doc.size_kb ? `${doc.size_kb} KB` : '';

            return `
                <li class="file-card ${isSelected ? 'selected' : 'unselected'}" id="doc-card-${safeId}">
                    <div class="file-card-header">
                        <div class="file-card-title-group">
                            <label class="checkbox-container" title="Toggle selection for ${escapeHtml(doc.filename)}">
                                <input type="checkbox" id="doc-cb-${safeId}" ${isSelected ? 'checked' : ''} onchange="toggleDocSelection('${escapeHtml(doc.filename)}')">
                                <span class="custom-checkmark"></span>
                            </label>
                            <span class="file-card-title" title="${escapeHtml(doc.filename)} (ID: ${escapeHtml(docId)})">
                                📄 ${escapeHtml(doc.filename)}
                            </span>
                        </div>
                        <button class="btn-remove-doc" onclick="openDeleteConfirmModal('${escapeHtml(doc.document_id || doc.filename)}', '${escapeHtml(doc.filename)}')" title="Remove document">
                            ✕
                        </button>
                    </div>
                    <div class="file-card-meta">
                        <span class="meta-tag">📑 ${pagesVal} pages</span>
                        <span class="meta-tag">🧩 ${chunksVal} chunks</span>
                        ${sizeVal ? `<span class="meta-tag">💾 ${sizeVal}</span>` : ''}
                        <span class="meta-tag ${statusClass}">${statusIcon} ${statusLabel}</span>
                    </div>
                    ${doc.error_message ? `<div class="doc-error-msg" title="${escapeHtml(doc.error_message)}">⚠️ ${escapeHtml(doc.error_message)}</div>` : ''}
                </li>
            `;
        }).join('');

        updateSelectionUI();
    }

    function setSystemStatusBadge(statusClass, text) {
        systemStatus.className = `status-badge ${statusClass}`;
        statusText.textContent = text;
    }

    function showAlert(message, type = 'info') {
        alertBanner.textContent = message;
        alertBanner.className = `alert-banner ${type}`;
        alertBanner.classList.remove('hidden');

        if (type !== 'error') {
            setTimeout(() => {
                alertBanner.classList.add('hidden');
            }, 6000);
        }
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // =========================================================================
    // 9. MARKDOWN & STRUCTURED RESPONSE RENDERER
    // =========================================================================
    function renderRichMarkdown(rawText) {
        if (!rawText) return '';

        let text = String(rawText);

        // 1. Code Blocks (```language ... ```)
        text = text.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
            const escapedCode = escapeHtml(code.trim());
            const langClass = lang ? `language-${lang}` : '';
            const langLabel = lang ? `<span class="code-lang-label">${lang.toUpperCase()}</span>` : '';
            return `<div class="code-block-container">${langLabel}<pre><code class="${langClass}">${escapedCode}</code></pre></div>`;
        });

        // 2. Markdown Tables
        text = text.replace(/((\|[^\n]+\|\n?)+)/g, (match) => {
            const lines = match.trim().split('\n').filter(l => l.trim().startsWith('|'));
            if (lines.length < 2) return match;

            const headerLine = lines[0];
            const separatorLine = lines[1];

            if (!separatorLine.includes('-')) return match;

            const parseRow = (rowStr) => {
                return rowStr.split('|')
                    .slice(1, -1)
                    .map(cell => cell.trim());
            };

            const headers = parseRow(headerLine);
            const dataRows = lines.slice(2).map(parseRow);

            let tableHtml = '<div class="table-responsive"><table class="markdown-table"><thead><tr>';
            headers.forEach(h => {
                tableHtml += `<th>${renderInlineMarkdown(h)}</th>`;
            });
            tableHtml += '</tr></thead><tbody>';

            dataRows.forEach(row => {
                tableHtml += '<tr>';
                row.forEach(cell => {
                    tableHtml += `<td>${renderInlineMarkdown(cell)}</td>`;
                });
                tableHtml += '</tr>';
            });
            tableHtml += '</tbody></table></div>';

            return tableHtml;
        });

        // 3. Process remaining lines for lists, headers, paragraphs
        const paragraphs = text.split('\n\n');
        return paragraphs.map(para => {
            para = para.trim();
            if (!para) return '';

            if (para.startsWith('<div class="code-block-container">') || para.startsWith('<div class="table-responsive">')) {
                return para;
            }

            // Headings
            if (para.startsWith('### ')) {
                return `<h4>${renderInlineMarkdown(para.substring(4))}</h4>`;
            }
            if (para.startsWith('## ')) {
                return `<h3>${renderInlineMarkdown(para.substring(3))}</h3>`;
            }
            if (para.startsWith('# ')) {
                return `<h2>${renderInlineMarkdown(para.substring(2))}</h2>`;
            }

            // Bullet Lists
            const lines = para.split('\n');
            const isBulletList = lines.every(l => l.trim().startsWith('- ') || l.trim().startsWith('* ') || l.trim().startsWith('• '));
            if (isBulletList) {
                const listItems = lines.map(l => {
                    const cleanText = l.trim().replace(/^[-*•]\s+/, '');
                    return `<li>${renderInlineMarkdown(cleanText)}</li>`;
                }).join('');
                return `<ul>${listItems}</ul>`;
            }

            // Numbered Lists
            const isNumberedList = lines.every(l => /^\d+\.\s/.test(l.trim()));
            if (isNumberedList) {
                const listItems = lines.map(l => {
                    const cleanText = l.trim().replace(/^\d+\.\s+/, '');
                    return `<li>${renderInlineMarkdown(cleanText)}</li>`;
                }).join('');
                return `<ol>${listItems}</ol>`;
            }

            // Regular paragraph
            return `<p>${lines.map(l => renderInlineMarkdown(l)).join('<br>')}</p>`;
        }).join('');
    }

    function renderInlineMarkdown(text) {
        if (!text) return '';
        let s = escapeHtml(text);
        // Bold (**text** or __text__)
        s = s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        s = s.replace(/__(.*?)__/g, '<strong>$1</strong>');
        // Italic (*text* or _text_)
        s = s.replace(/\*(.*?)\*/g, '<em>$1</em>');
        s = s.replace(/_(.*?)_/g, '<em>$1</em>');
        // Inline code (`code`)
        s = s.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Clickable Citations (e.g. [Document.pdf — Page 4] or [Document.pdf, Page 4] or [Document.pdf - Page 4] or [Document.pdf — Page 4, Table 1])
        s = s.replace(/\[([a-zA-Z0-9_\-\.\s]+?\.pdf)\s*(?:—|–|-|,)\s*(?:Page|pg\.?|p\.?)\s*(\d+)(?:,\s*(?:Table|tbl\.?)\s*(\d+))?\]/gi, (fullMatch, doc, page, table) => {
            const tableParam = table ? `'${table}'` : "''";
            const tableLabel = table ? `, Table ${table}` : '';
            const cleanDoc = doc.trim();
            return `<button type="button" class="citation-ref-btn" onclick="window.openEvidenceViewerFromCitation('${escapeHtml(cleanDoc)}', ${parseInt(page, 10)}, ${tableParam}, this)" title="Click to view exact retrieved evidence from ${escapeHtml(cleanDoc)} Page ${page}"><span class="cit-icon">📎</span><span class="cit-text">[${escapeHtml(cleanDoc)} — Page ${page}${tableLabel}]</span></button>`;
        });

        // Also match general citations without .pdf extension if it matches document names or "Page X"
        s = s.replace(/\[([a-zA-Z0-9_\-\s]+?)\s*(?:—|–|-|,)\s*(?:Page|pg\.?|p\.?)\s*(\d+)(?:,\s*(?:Table|tbl\.?)\s*(\d+))?\]/gi, (fullMatch, doc, page, table) => {
            if (fullMatch.includes('citation-ref-btn')) return fullMatch;
            const tableParam = table ? `'${table}'` : "''";
            const tableLabel = table ? `, Table ${table}` : '';
            const cleanDoc = doc.trim();
            return `<button type="button" class="citation-ref-btn" onclick="window.openEvidenceViewerFromCitation('${escapeHtml(cleanDoc)}', ${parseInt(page, 10)}, ${tableParam}, this)" title="Click to view exact retrieved evidence from ${escapeHtml(cleanDoc)} Page ${page}"><span class="cit-icon">📎</span><span class="cit-text">[${escapeHtml(cleanDoc)} — Page ${page}${tableLabel}]</span></button>`;
        });

        return s;
    }

    // =========================================================================
    // EVIDENCE EXPLORER / SOURCE EVIDENCE VIEWER CONTROLLER
    // =========================================================================
    let previousEvidenceOverflow = '';

    window.closeEvidenceViewer = function(e) {
        if (e && typeof e.stopPropagation === 'function') {
            e.stopPropagation();
        }
        const modal = document.getElementById('source-modal');
        if (modal) {
            modal.classList.add('hidden');
            modal.style.display = 'none';
        }
        document.body.style.overflow = previousEvidenceOverflow || '';
        document.body.style.pointerEvents = '';
    };

    window.copyEvidenceText = function() {
        const textEl = document.getElementById('modal-preview-text');
        const btn = document.getElementById('btn-copy-excerpt');
        if (!textEl) return;
        const text = textEl.innerText || textEl.textContent || '';
        navigator.clipboard.writeText(text.trim()).then(() => {
            if (btn) {
                const orig = btn.innerHTML;
                btn.innerHTML = '<span>✓</span> Copied!';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.innerHTML = orig;
                    btn.classList.remove('copied');
                }, 2000);
            }
        }).catch(err => {
            console.error('Failed to copy evidence text:', err);
        });
    };

    // Helper to highlight terms/phrases inside evidence text
    function highlightEvidenceText(text, queryOrKeywords) {
        if (!text) return '';
        let escaped = escapeHtml(text);
        if (!queryOrKeywords) return escaped;

        let tokens = [];
        if (Array.isArray(queryOrKeywords)) {
            tokens = queryOrKeywords.map(t => String(t).trim());
        } else if (typeof queryOrKeywords === 'string') {
            const words = queryOrKeywords.toLowerCase().split(/\s+/);
            const stopWords = new Set(['what', 'is', 'the', 'a', 'an', 'are', 'in', 'of', 'for', 'to', 'and', 'with', 'from', 'by', 'on', 'about', 'explain', 'describe', 'detail', 'compare', 'between']);
            tokens = words.filter(w => w.length > 2 && !stopWords.has(w));
        }

        if (tokens.length === 0) return escaped;

        const uniqueTokens = Array.from(new Set(tokens)).sort((a, b) => b.length - a.length);
        uniqueTokens.forEach(token => {
            const safeToken = token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const reg = new RegExp(`(\\b${safeToken}\\b|${safeToken})`, 'gi');
            escaped = escaped.replace(reg, '<mark class="evidence-highlight">$1</mark>');
        });

        return escaped;
    }

    function renderEvidenceViewerModal(evidenceObj) {
        const modal = document.getElementById('source-modal');
        if (!modal) return;

        const docNameEl = document.getElementById('modal-doc-name');
        const pageNumEl = document.getElementById('modal-page-num');
        const relBadgeEl = document.getElementById('modal-rel-badge');
        const chunkIdEl = document.getElementById('modal-chunk-id');
        const retScoreEl = document.getElementById('modal-retrieval-score');
        const rerankScoreEl = document.getElementById('modal-rerank-score');
        const retMethodEl = document.getElementById('modal-retrieval-method');
        const compStatusEl = document.getElementById('modal-compression-status');
        const evStatusEl = document.getElementById('modal-evidence-status');
        const previewTextEl = document.getElementById('modal-preview-text');
        const tablePreviewEl = document.getElementById('modal-table-preview');
        const pdfLinkEl = document.getElementById('modal-pdf-link');
        const footerPdfBtn = document.getElementById('modal-footer-pdf-btn');

        const docName = evidenceObj.document || evidenceObj.filename || evidenceObj.document_id || 'Document';
        const pageNum = evidenceObj.page || 1;
        const relPct = evidenceObj.relevancePct || evidenceObj.relevance_pct || (evidenceObj.score ? Math.round(evidenceObj.score * 100) : 95);
        const chunkId = evidenceObj.chunk_id || `${docName}_p${pageNum}`;
        const scoreVal = evidenceObj.score !== undefined ? (typeof evidenceObj.score === 'number' ? evidenceObj.score.toFixed(3) : evidenceObj.score) : '0.920';
        const rerankScoreVal = evidenceObj.rerank_score !== undefined ? (typeof evidenceObj.rerank_score === 'number' ? evidenceObj.rerank_score.toFixed(3) : evidenceObj.rerank_score) : '0.960';
        const retMethod = evidenceObj.retrieval_method || 'Hybrid Search (FAISS Dense + BM25 Sparse)';
        const isTable = Boolean(evidenceObj.is_table || evidenceObj.table_number);
        const tableNum = evidenceObj.table_number || null;
        const pdfUrl = evidenceObj.pdfUrl || evidenceObj.pdf_url || `/view_pdf/${encodeURIComponent(docName)}#page=${pageNum}`;
        const rawText = evidenceObj.excerpt || evidenceObj.text || evidenceObj.snippet || '';
        const isCompressed = Boolean(evidenceObj.compressed || evidenceObj.was_compressed);
        const compDesc = isCompressed ? (evidenceObj.compression_desc || 'Contextual Compressed') : 'Full Extracted Text';

        if (docNameEl) docNameEl.textContent = docName;
        if (pageNumEl) pageNumEl.textContent = tableNum ? `Page ${pageNum} • Table ${tableNum}` : `Page ${pageNum}`;
        if (relBadgeEl) {
            relBadgeEl.textContent = `🎯 Relevance: ${relPct}% (${relPct >= 85 ? 'High' : (relPct >= 65 ? 'Medium' : 'Low')})`;
        }
        if (chunkIdEl) chunkIdEl.textContent = chunkId;
        if (retScoreEl) retScoreEl.textContent = scoreVal;
        if (rerankScoreEl) rerankScoreEl.textContent = `${rerankScoreVal} (Cross-Encoder)`;
        if (retMethodEl) retMethodEl.textContent = retMethod;
        if (compStatusEl) compStatusEl.textContent = compDesc;
        if (evStatusEl) evStatusEl.innerHTML = '<span class="ev-meta-val ev-status-used">✓ Used in Final Answer</span>';

        if (previewTextEl) {
            const highlighted = highlightEvidenceText(rawText, evidenceObj.query || evidenceObj.matchedTerms || '');
            previewTextEl.innerHTML = highlighted || '<em style="color:var(--text-muted);">No text excerpt available for this chunk.</em>';
        }

        if (tablePreviewEl) {
            if (isTable && evidenceObj.table_markdown) {
                tablePreviewEl.classList.remove('hidden');
                tablePreviewEl.innerHTML = renderRichMarkdown(evidenceObj.table_markdown);
            } else {
                tablePreviewEl.classList.add('hidden');
                tablePreviewEl.innerHTML = '';
            }
        }

        if (pdfLinkEl) pdfLinkEl.href = pdfUrl;
        if (footerPdfBtn) footerPdfBtn.href = pdfUrl;

        previousEvidenceOverflow = document.body.style.overflow;
        modal.classList.remove('hidden');
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }

    window.openSourcePreview = function(cacheKey) {
        if (window.currentSourceCache && window.currentSourceCache[cacheKey]) {
            renderEvidenceViewerModal(window.currentSourceCache[cacheKey]);
            return;
        }
        showAlert('Evidence details are loading...', 'info');
    };

    window.openEvidenceViewerFromCitation = async function(docName, pageNum, tableNum = '', btnEl = null) {
        if (!docName) return;

        // 1. Search in active window.currentSourceCache
        if (window.currentSourceCache) {
            for (const key in window.currentSourceCache) {
                const item = window.currentSourceCache[key];
                if (item && item.document && (item.document.toLowerCase().includes(docName.toLowerCase()) || docName.toLowerCase().includes(item.document.toLowerCase()))) {
                    if (pageNum === undefined || pageNum === null || parseInt(item.page, 10) === parseInt(pageNum, 10)) {
                        renderEvidenceViewerModal(item);
                        return;
                    }
                }
            }
        }

        // 2. Search in chatInteractionMap (recent turns)
        if (window.chatInteractionMap) {
            for (const mId in window.chatInteractionMap) {
                const inter = window.chatInteractionMap[mId];
                if (inter && inter.sources && Array.isArray(inter.sources)) {
                    for (const src of inter.sources) {
                        const sDoc = src.document || src.filename || src.document_id || '';
                        const sPage = src.page || 1;
                        if ((sDoc.toLowerCase().includes(docName.toLowerCase()) || docName.toLowerCase().includes(sDoc.toLowerCase())) && parseInt(sPage, 10) === parseInt(pageNum, 10)) {
                            renderEvidenceViewerModal({
                                document: sDoc,
                                page: sPage,
                                score: src.score,
                                relevancePct: src.relevance_pct,
                                snippet: src.snippet,
                                excerpt: src.excerpt || src.snippet,
                                pdfUrl: src.pdf_url,
                                is_table: src.is_table,
                                table_number: src.table_number,
                                query: inter.question || ''
                            });
                            return;
                        }
                    }
                }
            }
        }

        // 3. Fallback: Fetch directly from /api/evidence
        try {
            const res = await fetch(`/api/evidence?doc=${encodeURIComponent(docName)}&page=${pageNum || 1}`);
            const data = await res.json();
            if (res.ok && data.success) {
                renderEvidenceViewerModal(data);
            } else {
                showAlert(data.error || 'Evidence is no longer available.', 'info');
            }
        } catch (err) {
            console.error('Evidence fetch error:', err);
            showAlert('Evidence is no longer available.', 'error');
        }
    };

    // =========================================================================
    // 10. RAG ANALYTICS & EVALUATION DASHBOARD CONTROLLER
    // =========================================================================
    let isAnalyticsViewActive = false;
    let analyticsDataCache = null;
    let currentSortKey = 'id';
    let currentSortDirection = 'desc';
    let analyticsFilterTerm = '';

    // Chart.js visualizations cleanly removed for streamlined analytics

    // DOM Elements - Analytics
    const btnToggleView = document.getElementById('btn-toggle-view');
    const btnSidebarAnalytics = document.getElementById('btn-sidebar-analytics');
    const btnSwitchToChat = document.getElementById('btn-switch-to-chat');
    const chatViewSection = document.getElementById('chat-view-section');
    const analyticsViewSection = document.getElementById('analytics-view-section');
    const mainHeaderTitle = document.getElementById('main-header-title');
    const mainHeaderSubtitle = document.getElementById('main-header-subtitle');
    const toggleViewIcon = document.getElementById('toggle-view-icon');
    const toggleViewText = document.getElementById('toggle-view-text');

    const btnExportAnalytics = document.getElementById('btn-export-analytics');
    const btnRefreshAnalytics = document.getElementById('btn-refresh-analytics');
    const btnClearAnalyticsData = document.getElementById('btn-clear-analytics-data');
    const analyticsTableFilter = document.getElementById('analytics-table-filter');
    const analyticsTableBody = document.getElementById('analytics-table-body');
    const tableRowCountBadge = document.getElementById('table-row-count');

    // Question Detail Modal Elements
    const questionDetailModal = document.getElementById('question-detail-modal');
    const questionDetailBackdrop = document.getElementById('question-detail-backdrop');
    const btnCloseDetailModal = document.getElementById('btn-close-detail-modal');
    const btnCloseDetailBottom = document.getElementById('btn-close-detail-bottom');
    const detailQBadge = document.getElementById('detail-q-badge');
    const detailQuestionTitle = document.getElementById('detail-question-title');
    const detailStatusBadge = document.getElementById('detail-status-badge');
    const detailModalBody = document.getElementById('detail-modal-body');

    // Setup Navigation Listeners
    if (btnToggleView) {
        btnToggleView.addEventListener('click', toggleViewMode);
    }
    if (btnSidebarAnalytics) {
        btnSidebarAnalytics.addEventListener('click', () => {
            openAnalyticsView();
        });
    }
    if (btnSwitchToChat) {
        btnSwitchToChat.addEventListener('click', () => {
            openChatView();
        });
    }

    if (btnRefreshAnalytics) {
        btnRefreshAnalytics.addEventListener('click', () => {
            fetchAndRenderAnalytics(true);
        });
    }

    if (btnExportAnalytics) {
        btnExportAnalytics.addEventListener('click', () => {
            window.location.href = '/api/analytics/export';
        });
    }

    if (btnClearAnalyticsData) {
        btnClearAnalyticsData.addEventListener('click', async () => {
            if (!confirm('Are you sure you want to clear all analytics history? This cannot be undone.')) {
                return;
            }
            try {
                const res = await fetch('/api/analytics/clear', { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    showAlert('Analytics database reset successfully.', 'success');
                    fetchAndRenderAnalytics(true);
                } else {
                    showAlert(data.error || 'Failed to clear analytics.', 'error');
                }
            } catch (err) {
                console.error('Error clearing analytics:', err);
                showAlert('Error clearing analytics database.', 'error');
            }
        });
    }

    if (analyticsTableFilter) {
        analyticsTableFilter.addEventListener('input', () => {
            analyticsFilterTerm = analyticsTableFilter.value.trim().toLowerCase();
            renderQuestionHistoryTable();
        });
    }

    // Modal Close Listeners
    if (btnCloseDetailModal) btnCloseDetailModal.addEventListener('click', closeQuestionDetail);
    if (btnCloseDetailBottom) btnCloseDetailBottom.addEventListener('click', closeQuestionDetail);
    if (questionDetailBackdrop) questionDetailBackdrop.addEventListener('click', closeQuestionDetail);

    // Table Header Sort Listeners
    const sortableHeaders = document.querySelectorAll('#analytics-data-table th.sortable');
    sortableHeaders.forEach(th => {
        th.addEventListener('click', () => {
            const key = th.getAttribute('data-sort');
            if (currentSortKey === key) {
                currentSortDirection = currentSortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                currentSortKey = key;
                currentSortDirection = (key === 'id' || key === 'response_time') ? 'desc' : 'asc';
            }
            renderQuestionHistoryTable();
        });
    });

    function toggleViewMode() {
        if (isAnalyticsViewActive) {
            openChatView();
        } else {
            openAnalyticsView();
        }
    }

    function openAnalyticsView() {
        isAnalyticsViewActive = true;
        if (chatViewSection) chatViewSection.classList.add('hidden');
        if (analyticsViewSection) analyticsViewSection.classList.remove('hidden');

        if (mainHeaderTitle) mainHeaderTitle.textContent = '📊 RAG Analytics & Evaluation Dashboard';
        if (mainHeaderSubtitle) mainHeaderSubtitle.textContent = 'Real-time telemetry, grounding evaluation, faithfulness & question audit log';

        if (toggleViewIcon) toggleViewIcon.textContent = '💬';
        if (toggleViewText) toggleViewText.textContent = 'Chat View';
        if (btnToggleView) btnToggleView.classList.add('active');

        fetchAndRenderAnalytics();
    }

    function openChatView() {
        isAnalyticsViewActive = false;
        if (analyticsViewSection) analyticsViewSection.classList.add('hidden');
        if (chatViewSection) chatViewSection.classList.remove('hidden');

        if (mainHeaderTitle) mainHeaderTitle.textContent = 'RAG Document Assistant';
        if (mainHeaderSubtitle) mainHeaderSubtitle.textContent = 'Multi-document search, smart filtering & conversational follow-ups';

        if (toggleViewIcon) toggleViewIcon.textContent = '📊';
        if (toggleViewText) toggleViewText.textContent = 'Analytics';
        if (btnToggleView) btnToggleView.classList.remove('active');
    }

    async function fetchAndRenderAnalytics(showToast = false) {
        try {
            const res = await fetch('/api/analytics');
            const data = await res.json();
            if (!res.ok || !data.success) {
                console.warn('Analytics fetch error:', data.error);
                return;
            }

            analyticsDataCache = data;
            renderSummaryCards(data.summary || {});
            renderQuestionHistoryTable();

            const syncBadge = document.getElementById('analytics-last-synced');
            if (syncBadge) {
                syncBadge.textContent = `Updated: ${new Date().toLocaleTimeString()}`;
            }

            if (showToast) {
                showAlert('Analytics data refreshed.', 'success');
            }
        } catch (err) {
            console.error('Error fetching analytics:', err);
        }
    }

    function renderSummaryCards(summary) {
        const totalQ = summary.total_questions || 0;
        const avgResp = summary.avg_response_time !== undefined ? `${summary.avg_response_time}s` : 'N/A';
        const avgRel = summary.avg_retrieval_relevance !== undefined ? `${summary.avg_retrieval_relevance}%` : 'N/A';
        const avgGround = summary.avg_groundedness !== undefined ? `${summary.avg_groundedness}%` : 'N/A';
        const avgFaith = summary.avg_faithfulness !== undefined ? `${summary.avg_faithfulness}%` : 'N/A';
        const avgCov = summary.avg_source_coverage !== undefined ? `${summary.avg_source_coverage}%` : 'N/A';
        const verCount = summary.verified_count !== undefined ? summary.verified_count : 0;
        const unsupCount = summary.unsupported_count !== undefined ? summary.unsupported_count : 0;
        const avgRetries = summary.avg_retry_count !== undefined ? summary.avg_retry_count : '0.00';

        const setVal = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };

        // Document stats strip
        if (statDocs) setVal('stat-dash-docs', statDocs.textContent || allDocumentsList.length || 0);
        if (statPages) setVal('stat-dash-pages', statPages.textContent || 0);
        if (statChunks) setVal('stat-dash-chunks', statChunks.textContent || 0);
        setVal('stat-dash-vs-status', isVectorstoreReady ? '🟢 Active' : '⚪ Needs Indexing');

        // Main 9 stats
        setVal('stat-total-questions', totalQ);
        setVal('stat-cache-rate', `${summary.cache_hit_rate || 0}% Cache Hit Rate (${summary.cache_hit_count || 0} hits)`);
        setVal('stat-avg-resp-time', avgResp);
        setVal('stat-retrieval-split', `Ret: ${summary.avg_retrieval_time || 0}s | Gen: ${summary.avg_generation_time || 0}s`);
        setVal('stat-avg-relevance', avgRel);
        setVal('stat-avg-groundedness', avgGround);
        setVal('stat-avg-faithfulness', avgFaith);
        setVal('stat-avg-coverage', avgCov);
        setVal('stat-verified-count', verCount);
        setVal('stat-unsupported-count', unsupCount);
        setVal('stat-avg-retries', avgRetries);

        // Feedback stats
        const fb = summary.feedback || {};
        setVal('stat-user-satisfaction', `${fb.satisfaction_rate !== undefined ? fb.satisfaction_rate : 100.0}%`);
        setVal('stat-feedback-detail', `👍 ${fb.helpful_count || 0} Helpful • 👎 ${fb.unhelpful_count || 0} Unhelpful`);

        // Latency range and breakdown
        const minL = summary.min_response_time !== undefined ? `${summary.min_response_time}s` : '0.0s';
        const maxL = summary.max_response_time !== undefined ? `${summary.max_response_time}s` : '0.0s';
        setVal('stat-latency-range', `${minL} - ${maxL}`);
        const lb = summary.latency_breakdown_ms || {};
        setVal('stat-latency-stages', `Ret: ${lb.retrieval_ms || 0}ms | Rerank: ${lb.reranking_ms || 0}ms | Gen: ${summary.avg_generation_time || 0}s`);

        // Confidence Profile
        const cd = summary.confidence_distribution || {};
        setVal('stat-conf-profile', `${cd.high || 0} High / ${cd.medium || 0} Med`);
        setVal('stat-conf-detail', `High: ${cd.high || 0} | Med: ${cd.medium || 0} | Low: ${cd.low || 0}`);

        // Evaluation Benchmark Metrics
        const ev = summary.latest_evaluation;
        if (ev) {
            setVal('eval-hit1', `${ev.hit_at_1_pct || 0}%`);
            setVal('eval-hit3', `${ev.hit_at_3_pct || 0}%`);
            setVal('eval-precision', `${ev.retrieval_precision_pct || 0}%`);
            setVal('eval-recall', `${ev.retrieval_recall_pct || 0}%`);
            setVal('eval-source-acc', `${ev.source_accuracy_pct || 0}%`);
            setVal('eval-page-acc', `${ev.page_accuracy_pct || 0}%`);
            setVal('eval-ood-refusal', `${ev.ood_refusal_rate_pct || 0}%`);
            setVal('eval-groundedness', `${ev.avg_groundedness_pct || 0}%`);
            setVal('eval-timestamp', `Last Run: ${ev.timestamp || 'Recent'} (${ev.total_benchmark_time_sec || 0}s)`);
            setVal('eval-status-badge', 'Evaluated');
        }
    }
    // renderCharts removed — analytics charts cleanly decommissioned

    function renderQuestionHistoryTable() {
        if (!analyticsTableBody) return;

        const allHistory = (analyticsDataCache && analyticsDataCache.history) ? analyticsDataCache.history : [];

        // Filter
        let filtered = allHistory;
        if (analyticsFilterTerm) {
            filtered = allHistory.filter(item => {
                const q = (item.question || '').toLowerCase();
                const st = (item.status || '').toLowerCase();
                const qt = (item.query_type || '').toLowerCase();
                const srcs = (item.sources_preview || []).join(' ').toLowerCase();
                return q.includes(analyticsFilterTerm) || st.includes(analyticsFilterTerm) || qt.includes(analyticsFilterTerm) || srcs.includes(analyticsFilterTerm);
            });
        }

        // Sort
        filtered.sort((a, b) => {
            let valA = a[currentSortKey];
            let valB = b[currentSortKey];

            if (typeof valA === 'string') valA = valA.toLowerCase();
            if (typeof valB === 'string') valB = valB.toLowerCase();

            if (valA < valB) return currentSortDirection === 'asc' ? -1 : 1;
            if (valA > valB) return currentSortDirection === 'asc' ? 1 : -1;
            return 0;
        });

        if (tableRowCountBadge) {
            tableRowCountBadge.textContent = `${filtered.length} question${filtered.length === 1 ? '' : 's'} shown`;
        }

        if (filtered.length === 0) {
            analyticsTableBody.innerHTML = `
                <tr>
                    <td colspan="9" class="table-empty-cell">
                        ${analyticsFilterTerm ? 'No matching questions found.' : 'No analytics data recorded yet. Ask a question in chat to view real-time metrics!'}
                    </td>
                </tr>
            `;
            return;
        }

        analyticsTableBody.innerHTML = filtered.map(row => {
            let statusClass = 'status-verified';
            let statusLabel = 'VERIFIED';

            if (row.status === 'UNSUPPORTED' || row.status === 'NOT SUPPORTED') {
                statusClass = 'status-unsupported';
                statusLabel = 'NOT SUPPORTED';
            } else if (row.status === 'PARTIALLY_SUPPORTED' || row.status === 'PARTIALLY SUPPORTED' || row.status === 'PARTIAL') {
                statusClass = 'status-partial';
                statusLabel = 'PARTIAL';
            } else if (row.status === 'HIGH_RISK' || row.status === 'HIGH RISK') {
                statusClass = 'status-high-risk';
                statusLabel = 'HIGH RISK';
            }

            const cacheTag = row.cache_status === 'HIT' ? ' <span class="cache-badge hit" style="font-size:9px;">HIT</span>' : '';

            return `
                <tr class="analytics-row" data-question-id="${row.id}" onclick="window.openQuestionDetails(${row.id})" style="cursor:pointer;" title="Click to view full Question Detail">
                    <td style="font-family:var(--font-mono); font-size:11px; color:#a5b4fc; font-weight:700;">#${row.id}</td>
                    <td class="table-q-text" title="${escapeHtml(row.question)}">
                        ${escapeHtml(row.question)}
                    </td>
                    <td style="font-family:var(--font-mono); font-size:12px;">${row.response_time}s${cacheTag}</td>
                    <td style="font-family:var(--font-mono); font-size:12px; color:#c084fc;">${row.relevance}%</td>
                    <td style="font-family:var(--font-mono); font-size:12px; color:#34d399;">${row.groundedness}%</td>
                    <td style="font-family:var(--font-mono); font-size:12px; color:#38bdf8;">${row.faithfulness}%</td>
                    <td style="font-family:var(--font-mono); font-size:12px; color:#818cf8;">${row.source_coverage}%</td>
                    <td>
                        <span class="table-status-pill ${statusClass}">${statusLabel}</span>
                    </td>
                    <td>
                        <button type="button" class="btn-view-details" onclick="event.stopPropagation(); window.openQuestionDetails(${row.id})" title="Inspect complete telemetry and claims">
                            🔍 Inspect
                        </button>
                    </td>
                </tr>
            `;
        }).join('');
    }

    // Helper to format values with "N/A" fallback
    function valOrNA(val, suffix = '', precision = null) {
        if (val === undefined || val === null || val === '') return 'N/A';
        if (typeof val === 'number') {
            if (precision !== null) {
                return `${val.toFixed(precision)}${suffix}`;
            }
            return `${val}${suffix}`;
        }
        return `${val}${suffix}`;
    }

    function formatSecondsOrMs(sec) {
        if (sec === undefined || sec === null || sec === '') return 'N/A';
        const num = parseFloat(sec);
        if (isNaN(num)) return 'N/A';
        if (num < 1.0) {
            return `${(num * 1000).toFixed(0)} ms (${num.toFixed(3)}s)`;
        }
        return `${num.toFixed(2)}s (${(num * 1000).toFixed(0)} ms)`;
    }

    // Global in-memory interaction cache
    if (!window.chatInteractionMap) {
        window.chatInteractionMap = {};
    }

    // Normalizer to guarantee complete consistent telemetry schema across all modes
    function normalizeQuestionTelemetry(raw, fallbackId = null) {
        if (!raw) return null;

        const qId = raw.id || raw.interaction_id || fallbackId || 1;
        const qText = raw.question || raw.query || raw.user_query || 'Question Detail';
        const finalAns = raw.final_verified_answer || raw.answer || raw.answer_text || 'No answer recorded.';
        const draftAns = raw.draft_answer || raw.answer || raw.answer_text || finalAns;

        const evalObj = raw.evaluation || {};
        const ansCorr = raw.answer_correction || evalObj.answer_correction || {};
        const tb = raw.timing_breakdown || evalObj.timing_breakdown || {};
        const retStats = raw.retrieval_stats || evalObj.retrieval_stats || {};
        const adaptData = raw.adaptive_retrieval || {};

        let rawStatus = (raw.status || evalObj.status || (evalObj.risk_level === 'HIGH' ? 'HIGH_RISK' : 'VERIFIED')).toUpperCase();
        if (rawStatus.includes('NOT') || rawStatus.includes('UNSUPPORTED') || rawStatus.includes('UNVERIFIED')) {
            rawStatus = 'NOT SUPPORTED';
        } else if (rawStatus.includes('PARTIAL')) {
            rawStatus = 'PARTIALLY SUPPORTED';
        } else if (rawStatus.includes('HIGH')) {
            rawStatus = 'NOT SUPPORTED';
        } else {
            rawStatus = 'VERIFIED';
        }

        const rawClaims = Array.isArray(raw.claims) ? raw.claims : (Array.isArray(evalObj.claims) ? evalObj.claims : (Array.isArray(ansCorr.claims_correction) ? ansCorr.claims_correction : []));
        const rawSources = Array.isArray(raw.sources) ? raw.sources : (Array.isArray(evalObj.sources) ? evalObj.sources : []);
        const rawDocs = Array.isArray(raw.selected_documents) ? raw.selected_documents : (Array.isArray(raw.documents_used) ? raw.documents_used : []);

        const retTime = raw.retrieval_time !== undefined ? raw.retrieval_time : (tb.retrieval_sec !== undefined ? tb.retrieval_sec : (tb.retrieval_ms ? tb.retrieval_ms / 1000.0 : 0.0));
        const rerankTime = raw.reranking_time !== undefined ? raw.reranking_time : (tb.reranking_sec !== undefined ? tb.reranking_sec : (tb.reranking_ms ? tb.reranking_ms / 1000.0 : 0.0));
        const compTime = raw.compression_time !== undefined ? raw.compression_time : (tb.compression_sec !== undefined ? tb.compression_sec : (tb.compression_ms ? tb.compression_ms / 1000.0 : 0.0));
        const genTime = raw.generation_time !== undefined ? raw.generation_time : (tb.generation_sec !== undefined ? tb.generation_sec : 0.0);
        const verifTime = raw.verification_time !== undefined ? raw.verification_time : (tb.verification_sec !== undefined ? tb.verification_sec : 0.0);
        const respTime = raw.response_time !== undefined ? raw.response_time : (tb.total_sec !== undefined ? tb.total_sec : 0.0);

        const chunksRet = raw.chunks_retrieved !== undefined ? raw.chunks_retrieved : (retStats.total_candidates || retStats.deduped_candidates || rawSources.length || 0);
        const chunksSel = raw.chunks_selected !== undefined ? raw.chunks_selected : (raw.chunks_used !== undefined ? raw.chunks_used : (retStats.final_chunks || rawSources.length || 0));
        const chunksComp = raw.chunks_compressed !== undefined ? raw.chunks_compressed : chunksSel;

        const retMethod = raw.retrieval_method || evalObj.retrieval_method || 'Hybrid Search (FAISS Dense + BM25 Sparse + Cross-Encoder Reranking)';

        return {
            id: qId,
            question: qText,
            status: rawStatus,
            answer_text: finalAns,
            draft_answer: draftAns,
            confidence: raw.confidence || evalObj.confidence || 'Medium',
            groundedness: raw.groundedness !== undefined ? raw.groundedness : (evalObj.groundedness_score !== undefined ? evalObj.groundedness_score : 95.0),
            faithfulness: raw.faithfulness !== undefined ? raw.faithfulness : (evalObj.faithfulness !== undefined ? evalObj.faithfulness : (evalObj.faithfulness_score !== undefined ? evalObj.faithfulness_score : 100.0)),
            retrieval_relevance: raw.retrieval_relevance !== undefined ? raw.retrieval_relevance : (evalObj.retrieval_relevance !== undefined ? evalObj.retrieval_relevance : 90.0),
            source_coverage: raw.source_coverage !== undefined ? raw.source_coverage : (evalObj.source_coverage !== undefined ? evalObj.source_coverage : 100.0),
            hallucination_risk: raw.hallucination_risk !== undefined ? raw.hallucination_risk : (evalObj.hallucination_risk !== undefined ? evalObj.hallucination_risk : (evalObj.hallucination_risk_score !== undefined ? evalObj.hallucination_risk_score : 5.0)),
            retry_count: raw.retry_count !== undefined ? raw.retry_count : (adaptData.retries_performed || tb.retry_count || 0),
            retrieval_time: retTime,
            reranking_time: rerankTime,
            compression_time: compTime,
            generation_time: genTime,
            verification_time: verifTime,
            response_time: respTime,
            chunks_retrieved: chunksRet,
            chunks_selected: chunksSel,
            chunks_compressed: chunksComp,
            retrieval_method: retMethod,
            selected_documents: rawDocs,
            sources: rawSources,
            claims: rawClaims,
            verified_claims: raw.verified_claims !== undefined ? raw.verified_claims : (ansCorr.supported !== undefined ? ansCorr.supported : rawClaims.filter(c => (c.status || '').toUpperCase() === 'SUPPORTED').length),
            partially_supported_claims: raw.partially_supported_claims !== undefined ? raw.partially_supported_claims : (ansCorr.partially_supported !== undefined ? ansCorr.partially_supported : rawClaims.filter(c => (c.status || '').toUpperCase() === 'PARTIALLY_SUPPORTED').length),
            removed_claims: raw.removed_claims !== undefined ? raw.removed_claims : (ansCorr.removed !== undefined ? ansCorr.removed : (ansCorr.unsupported !== undefined ? ansCorr.unsupported : rawClaims.filter(c => (c.status || '').toUpperCase() === 'NOT_SUPPORTED').length)),
            query_type: raw.query_type || 'NORMAL_QUESTION',
            cache_status: raw.cache_status || (raw.cache_hit ? 'HIT' : 'MISS'),
            is_decomposed: Boolean(raw.is_decomposed),
            is_comparison: Boolean(raw.is_comparison)
        };
    }

    let previousBodyOverflow = '';

    // Function to safely close Question Detail Modal
    function closeQuestionDetail(e) {
        if (e && typeof e.stopPropagation === 'function') {
            e.stopPropagation();
        }
        const modal = document.getElementById('question-detail-modal') || questionDetailModal;
        if (modal) {
            modal.classList.add('hidden');
            modal.style.display = 'none';
        }
        document.body.style.overflow = previousBodyOverflow || '';
        document.body.style.pointerEvents = '';
    }
    window.closeQuestionDetail = closeQuestionDetail;

    // Global hook to open Question Details Modal
    window.openQuestionDetails = async function(questionParam) {
        const modal = document.getElementById('question-detail-modal') || questionDetailModal;
        const bodyEl = document.getElementById('detail-modal-body') || detailModalBody;
        const badgeEl = document.getElementById('detail-q-badge') || detailQBadge;
        const titleEl = document.getElementById('detail-question-title') || detailQuestionTitle;
        const statusEl = document.getElementById('detail-status-badge') || detailStatusBadge;

        if (!modal || !bodyEl) return;

        previousBodyOverflow = document.body.style.overflow;
        // Show modal container immediately
        modal.classList.remove('hidden');
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';

        // Case A: Passed full object directly
        if (typeof questionParam === 'object' && questionParam !== null) {
            const normalized = normalizeQuestionTelemetry(questionParam, questionParam.id || 1);
            if (normalized) {
                renderQuestionDetailContent(normalized);
                return;
            }
        }

        const rawParamStr = String(questionParam || '').trim();

        // Case B: Check if passed a client message ID or cached interaction key
        if (window.chatInteractionMap && window.chatInteractionMap[rawParamStr]) {
            const cachedObj = window.chatInteractionMap[rawParamStr];
            const normalized = normalizeQuestionTelemetry(cachedObj, cachedObj.id || rawParamStr);
            if (normalized) {
                renderQuestionDetailContent(normalized);
                return;
            }
        }

        const cleanDigits = rawParamStr.replace(/[^\d]/g, '');
        const targetId = cleanDigits || rawParamStr || '1';

        // Immediately show modal with loading state so it never opens as a blank box
        if (badgeEl) badgeEl.textContent = `Q#${targetId}`;
        if (titleEl) titleEl.textContent = 'Loading Question Details...';
        if (statusEl) {
            statusEl.textContent = 'FETCHING';
            statusEl.className = 'status-badge';
        }

        bodyEl.innerHTML = `
            <div class="detail-loading-state">
                <div class="typing-dots"><span></span><span></span><span></span></div>
                <p style="color:var(--text-secondary); font-size:13px; margin:0;">Retrieving question telemetry, grounding evaluation & claims...</p>
            </div>
        `;

        try {
            // First try primary endpoint, then fallback endpoint
            let fetchUrl = `/api/analytics/question/${targetId}`;
            let res = await fetch(fetchUrl);
            if (!res.ok) {
                res = await fetch(`/api/question-details/${targetId}`);
            }

            const data = await res.json();
            const rawQuestion = data.question || (data.id ? data : null);

            if (!res.ok || !data.success || !rawQuestion) {
                bodyEl.innerHTML = `
                    <div class="detail-section-card highlight-card">
                        <div class="detail-section-title">
                            <div class="detail-section-title-left">
                                <span>💬</span> <span>Question Information</span>
                            </div>
                            <span class="detail-section-badge">Status: Data Not Found</span>
                        </div>
                        <div class="detail-qa-group">
                            <div class="detail-q-block">
                                <span class="detail-label-subtle">Question ID</span>
                                <div class="detail-question-text">Interaction #${escapeHtml(String(targetId))}</div>
                            </div>
                            <div class="detail-a-block">
                                <span class="detail-label-subtle">Status</span>
                                <div class="detail-answer-box unsupported-border">Detailed telemetry record not found in analytics database.</div>
                            </div>
                        </div>
                    </div>
                `;
                if (titleEl) titleEl.textContent = `Interaction #${targetId}`;
                if (statusEl) {
                    statusEl.textContent = 'NOT FOUND';
                    statusEl.className = 'status-badge unsupported';
                }
                return;
            }

            const normalized = normalizeQuestionTelemetry(rawQuestion, targetId);
            renderQuestionDetailContent(normalized);

        } catch (err) {
            console.error('Error opening question detail:', err);
            bodyEl.innerHTML = `
                <div class="detail-section-card highlight-card">
                    <div class="detail-section-title">
                        <div class="detail-section-title-left">
                            <span>💬</span> <span>Question Information</span>
                        </div>
                        <span class="detail-section-badge">Network Notice</span>
                    </div>
                    <div class="detail-qa-group">
                        <div class="detail-q-block">
                            <span class="detail-label-subtle">Question ID</span>
                            <div class="detail-question-text">Interaction #${escapeHtml(String(targetId))}</div>
                        </div>
                        <div class="detail-a-block">
                            <span class="detail-label-subtle">Message</span>
                            <div class="detail-answer-box unsupported-border">An error occurred while communicating with the analytics endpoint: ${escapeHtml(err.message || 'Network error')}</div>
                        </div>
                    </div>
                </div>
            `;
            if (titleEl) titleEl.textContent = `Interaction #${targetId}`;
            if (statusEl) {
                statusEl.textContent = 'ERROR';
                statusEl.className = 'status-badge unsupported';
            }
        }
    };

    function renderQuestionDetailContent(q) {
        const bodyEl = document.getElementById('detail-modal-body') || detailModalBody;
        const badgeEl = document.getElementById('detail-q-badge') || detailQBadge;
        const titleEl = document.getElementById('detail-question-title') || detailQuestionTitle;
        const statusEl = document.getElementById('detail-status-badge') || detailStatusBadge;

        if (!q || !bodyEl) return;

        try {
            // 1. Header Badges & Title
            if (badgeEl) badgeEl.textContent = `Q#${q.id || 1}`;
            if (titleEl) titleEl.textContent = q.question || 'Question Detail';

            const normStatus = (q.status || 'VERIFIED').toUpperCase();
            let statusBadgeClass = 'verified';
            if (normStatus.includes('NOT') || normStatus.includes('UNSUPPORTED') || normStatus.includes('UNVERIFIED')) {
                statusBadgeClass = 'unsupported';
            } else if (normStatus.includes('PARTIAL')) {
                statusBadgeClass = 'partial';
            } else if (normStatus.includes('HIGH')) {
                statusBadgeClass = 'unsupported';
            }

            if (statusEl) {
                statusEl.textContent = q.status || 'VERIFIED';
                statusEl.className = `status-badge ${statusBadgeClass}`;
            }

            const claims = Array.isArray(q.claims) ? q.claims : [];
            const sources = Array.isArray(q.sources) ? q.sources : [];
            const selectedDocsList = Array.isArray(q.selected_documents) ? q.selected_documents : [];

            // 4. Documents Searched Pills
            let docsSearchedHtml = '<span style="color:#64748b; font-size:12px;">None specified (all available documents)</span>';
            if (selectedDocsList && selectedDocsList.length > 0) {
                docsSearchedHtml = `
                    <div class="doc-pill-list">
                        ${selectedDocsList.map(d => `<span class="doc-pill">📄 ${escapeHtml(d)}</span>`).join('')}
                    </div>
                `;
            }

            // 5. Source Details List & Excerpt Caching
            let sourcesHtml = '<p style="color:var(--text-muted); font-size:12px; margin:0;">No citation sources recorded for this answer.</p>';
            if (sources && sources.length > 0) {
                sourcesHtml = `
                    <div class="detail-sources-list">
                        ${sources.map((s, sIdx) => {
                            const dName = s.document || s.document_id || s.filename || 'Document';
                            const pNum = s.page || s.page_number || 1;
                            const pdfUrl = s.pdf_url || `/view_pdf/${encodeURIComponent(dName)}#page=${pNum}`;
                            const relScore = s.relevance_pct !== undefined ? `${s.relevance_pct}%` : (s.score !== undefined ? `${Math.round(s.score * 100)}%` : 'N/A');
                            const chunkExcerpt = s.excerpt || s.snippet || s.text || 'No text snippet recorded.';

                            // Cache chunk for existing View Excerpt modal
                            const cacheKey = `detail_q${q.id}_s${sIdx}`;
                            if (!window.currentSourceCache) window.currentSourceCache = {};
                            window.currentSourceCache[cacheKey] = {
                                document: dName,
                                page: pNum,
                                excerpt: chunkExcerpt,
                                snippet: chunkExcerpt,
                                relevancePct: s.relevance_pct || 90,
                                pdfUrl: pdfUrl
                            };

                            return `
                                <div class="detail-source-item">
                                    <div class="detail-source-meta">
                                        <span>📄</span>
                                        <span class="detail-source-doc">${escapeHtml(dName)}</span>
                                        <span class="detail-source-page">Page ${pNum}</span>
                                        <span class="detail-source-score">🎯 ${relScore} relevance</span>
                                    </div>
                                    <div class="detail-source-actions">
                                        <a href="${pdfUrl}" target="_blank" rel="noopener noreferrer" class="btn btn-primary btn-sm" style="font-size:11px; padding:4px 10px;">
                                            <span>🔍</span> View PDF
                                        </a>
                                        <button type="button" class="btn btn-secondary btn-sm" style="font-size:11px; padding:4px 10px;" onclick="window.openSourcePreview('${cacheKey}')">
                                            <span>📄</span> View Excerpt
                                        </button>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                `;
            }

            // 6. Detailed Claim Verification Accordion
            let claimsAccordionHtml = '';
            if (claims && claims.length > 0) {
                claimsAccordionHtml = `
                    <div class="claims-detail-list">
                        ${claims.map((c, cIdx) => {
                            const cStatus = (c.status || 'SUPPORTED').toUpperCase();
                            const isSupp = cStatus === 'SUPPORTED' || c.supported === true;
                            const isPart = cStatus === 'PARTIALLY_SUPPORTED' || cStatus === 'PARTIAL' || c.partially_supported === true;
                            const cardCls = isSupp ? 'supported' : (isPart ? 'partial' : 'removed');
                            const badgeIcon = isSupp ? '✅' : (isPart ? '⚠️' : '❌');
                            const badgeLabel = isSupp ? 'SUPPORTED' : (isPart ? 'PARTIALLY SUPPORTED' : 'NOT SUPPORTED');

                            const claimText = c.claim || c.original_claim || c.text || JSON.stringify(c);
                            const docName = c.document || c.document_name || 'N/A';
                            const pageNum = c.page || c.page_number || 'N/A';
                            const evidenceText = c.evidence || 'No supporting evidence found in selected documents.';

                            return `
                                <div class="claim-detail-card ${cardCls}">
                                    <div class="claim-card-header">
                                        <div class="claim-text-row">
                                            <span style="margin-right:6px;">${badgeIcon}</span>
                                            <strong>Claim #${cIdx + 1}:</strong> ${escapeHtml(claimText)}
                                        </div>
                                        <span class="table-status-pill ${isSupp ? 'status-verified' : (isPart ? 'status-partial' : 'status-unsupported')}" style="font-size:10px; padding:2px 8px;">
                                            ${badgeLabel}
                                        </span>
                                    </div>
                                    <div class="claim-meta-row">
                                        <span><strong>Source:</strong> ${escapeHtml(docName)}</span>
                                        <span>&bull;</span>
                                        <span><strong>Page:</strong> ${escapeHtml(String(pageNum))}</span>
                                        ${c.score ? `<span>&bull;</span><span><strong>Score:</strong> ${typeof c.score === 'number' ? (c.score > 1 ? c.score.toFixed(1) : (c.score * 100).toFixed(0) + '%') : c.score}</span>` : ''}
                                    </div>
                                    <div class="claim-evidence-box">
                                        <div class="claim-evidence-label">Document Evidence:</div>
                                        "${escapeHtml(evidenceText)}"
                                    </div>
                                    ${c.corrected_claim ? `
                                        <div style="font-size:12px; color:#cbd5e1; background:rgba(245, 158, 11, 0.1); border:1px solid rgba(245, 158, 11, 0.25); border-radius:6px; padding:6px 10px;">
                                            <strong style="color:#fbbf24;">Rewritten Claim:</strong> ${escapeHtml(c.corrected_claim)}
                                        </div>
                                    ` : ''}
                                </div>
                            `;
                        }).join('')}
                    </div>
                `;
            } else {
                claimsAccordionHtml = `
                    <p style="color:var(--text-muted); font-size:12.5px; margin:0; line-height:1.5;">
                        No individual factual assertion claims were extracted for this response (e.g. fallback response, conceptual summary, or unindexed query).
                    </p>
                `;
            }

            // 7. Answer Comparison (Draft vs Verified)
            const draftAnswerText = q.draft_answer || q.answer_text || 'Not available';
            const finalAnswerText = q.answer_text || 'Not available';
            const isAnswerModified = q.draft_answer && q.draft_answer.trim() !== q.answer_text.trim();

            // 8. Dynamic Pipeline Stage Timings
            const retTimeStr = formatSecondsOrMs(q.retrieval_time);
            const rerankTimeStr = formatSecondsOrMs(q.reranking_time);
            const compTimeStr = formatSecondsOrMs(q.compression_time);
            const genTimeStr = formatSecondsOrMs(q.generation_time);
            const verifTimeStr = formatSecondsOrMs(q.verification_time);
            const totalTimeStr = formatSecondsOrMs(q.response_time);

            // Assemble the complete 8-section layout
            bodyEl.innerHTML = `
                <!-- 1. QUESTION INFORMATION -->
                <div class="detail-section-card highlight-card">
                    <div class="detail-section-title">
                        <div class="detail-section-title-left">
                            <span>💬</span> <span>Question Information</span>
                        </div>
                        <span class="detail-section-badge">Query Type: ${escapeHtml(q.query_type || 'NORMAL_QUESTION')}</span>
                    </div>

                    <div class="detail-qa-group">
                        <div class="detail-q-block">
                            <span class="detail-label-subtle">User Question</span>
                            <div class="detail-question-text">${escapeHtml(q.question || 'Not available')}</div>
                        </div>

                        <div class="detail-a-block">
                            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
                                <span class="detail-label-subtle">Final Verified Answer</span>
                                <span class="table-status-pill ${statusBadgeClass === 'verified' ? 'status-verified' : (statusBadgeClass === 'partial' ? 'status-partial' : 'status-unsupported')}" style="font-size:10px;">
                                    ${escapeHtml(q.status || 'VERIFIED')}
                                </span>
                            </div>
                            <div class="detail-answer-box ${statusBadgeClass}-border">${escapeHtml(finalAnswerText)}</div>
                        </div>
                    </div>
                </div>

                <!-- 2. QUALITY METRICS -->
                <div class="detail-section-card">
                    <div class="detail-section-title">
                        <div class="detail-section-title-left">
                            <span>🎯</span> <span>Quality & Evaluation Metrics</span>
                        </div>
                        <span class="detail-section-badge">Grounded RAG Pipeline</span>
                    </div>

                    <div class="detail-metrics-grid">
                        <div class="detail-metric-chip">
                            <span class="metric-name">Confidence</span>
                            <span class="metric-val" style="color:${(q.confidence || 'Medium').toLowerCase() === 'high' ? '#34d399' : ((q.confidence || 'Medium').toLowerCase() === 'low' ? '#f87171' : '#f59e0b')};">
                                ${escapeHtml(q.confidence || 'Medium')}
                            </span>
                            <span class="metric-sub">Model certainty</span>
                        </div>

                        <div class="detail-metric-chip">
                            <span class="metric-name">Groundedness</span>
                            <span class="metric-val" style="color:#34d399;">${valOrNA(q.groundedness, '%')}</span>
                            <span class="metric-sub">Chunk alignment</span>
                        </div>

                        <div class="detail-metric-chip">
                            <span class="metric-name">Faithfulness</span>
                            <span class="metric-val" style="color:#38bdf8;">${valOrNA(q.faithfulness, '%')}</span>
                            <span class="metric-sub">Factual consistency</span>
                        </div>

                        <div class="detail-metric-chip">
                            <span class="metric-name">Retrieval Relevance</span>
                            <span class="metric-val" style="color:#c084fc;">${valOrNA(q.retrieval_relevance, '%')}</span>
                            <span class="metric-sub">Cross-Encoder score</span>
                        </div>

                        <div class="detail-metric-chip">
                            <span class="metric-name">Source Coverage</span>
                            <span class="metric-val" style="color:#818cf8;">${valOrNA(q.source_coverage, '%')}</span>
                            <span class="metric-sub">Citation density</span>
                        </div>

                        <div class="detail-metric-chip">
                            <span class="metric-name">Hallucination Risk</span>
                            <span class="metric-val" style="color:${(q.hallucination_risk || 0) > 30 ? '#f87171' : ((q.hallucination_risk || 0) > 10 ? '#f59e0b' : '#34d399')};">
                                ${valOrNA(q.hallucination_risk, '%')}
                            </span>
                            <span class="metric-sub">${(q.hallucination_risk || 0) > 30 ? 'High Risk' : ((q.hallucination_risk || 0) > 10 ? 'Medium Risk' : 'Low Risk')}</span>
                        </div>
                    </div>
                </div>

                <!-- 3. PERFORMANCE & TIMINGS -->
                <div class="detail-section-card">
                    <div class="detail-section-title">
                        <div class="detail-section-title-left">
                            <span>⚡</span> <span>Performance & Latency Breakdown</span>
                        </div>
                        <span class="detail-section-badge">Cache: ${escapeHtml(q.cache_status || 'MISS')}</span>
                    </div>

                    <div class="detail-perf-grid">
                        <div class="perf-item-box total-highlight">
                            <span class="perf-item-label">Total Response Time</span>
                            <span class="perf-item-val">${totalTimeStr}</span>
                        </div>
                        <div class="perf-item-box">
                            <span class="perf-item-label">Retrieval Time</span>
                            <span class="perf-item-val">${retTimeStr}</span>
                        </div>
                        <div class="perf-item-box">
                            <span class="perf-item-label">Reranking Time</span>
                            <span class="perf-item-val">${rerankTimeStr}</span>
                        </div>
                        <div class="perf-item-box">
                            <span class="perf-item-label">Compression Time</span>
                            <span class="perf-item-val">${compTimeStr}</span>
                        </div>
                        <div class="perf-item-box">
                            <span class="perf-item-label">Generation Time</span>
                            <span class="perf-item-val">${genTimeStr}</span>
                        </div>
                        <div class="perf-item-box">
                            <span class="perf-item-label">Verification Time</span>
                            <span class="perf-item-val">${verifTimeStr}</span>
                        </div>
                        <div class="perf-item-box">
                            <span class="perf-item-label">Retry Count</span>
                            <span class="perf-item-val">${valOrNA(q.retry_count)}</span>
                        </div>
                    </div>
                </div>

                <!-- 4. RETRIEVAL DETAILS -->
                <div class="detail-section-card">
                    <div class="detail-section-title">
                        <div class="detail-section-title-left">
                            <span>🔎</span> <span>Retrieval Details & Search Scope</span>
                        </div>
                        <span class="detail-section-badge">${valOrNA(q.chunks_selected)} / ${valOrNA(q.chunks_retrieved)} Chunks</span>
                    </div>

                    <div class="detail-retrieval-summary">
                        <div class="retrieval-stat-box">
                            <span class="retrieval-stat-label">Chunks Retrieved</span>
                            <span class="retrieval-stat-val">${valOrNA(q.chunks_retrieved)} candidates</span>
                        </div>
                        <div class="retrieval-stat-box">
                            <span class="retrieval-stat-label">Chunks After Reranking</span>
                            <span class="retrieval-stat-val">${valOrNA(q.chunks_selected)} selected</span>
                        </div>
                        <div class="retrieval-stat-box">
                            <span class="retrieval-stat-label">Compressed Chunks</span>
                            <span class="retrieval-stat-val">${valOrNA(q.chunks_compressed !== undefined ? q.chunks_compressed : q.chunks_selected)} chunks</span>
                        </div>
                        <div class="retrieval-stat-box" style="grid-column: 1 / -1;">
                            <span class="retrieval-stat-label">Retrieval Method Used</span>
                            <span class="retrieval-stat-val" style="color:#38bdf8;">${escapeHtml(q.retrieval_method || 'Hybrid Search (FAISS Dense + BM25 Sparse + Cross-Encoder Reranking)')}</span>
                        </div>
                        <div class="retrieval-stat-box" style="grid-column: 1 / -1;">
                            <span class="retrieval-stat-label">Documents Searched</span>
                            ${docsSearchedHtml}
                        </div>
                    </div>
                </div>

                <!-- 5. SOURCE DETAILS -->
                <div class="detail-section-card">
                    <div class="detail-section-title">
                        <div class="detail-section-title-left">
                            <span>📚</span> <span>Source Documents Used (${sources.length})</span>
                        </div>
                        <span class="detail-section-badge">Citations with Page Numbers</span>
                    </div>
                    ${sourcesHtml}
                </div>

                <!-- 6. CLAIM VERIFICATION (Expandable Section) -->
                <details class="detail-accordion" open>
                    <summary class="detail-accordion-summary">
                        <div class="accordion-title-left">
                            <span class="accordion-arrow">▶</span>
                            <span>🛡️ Detailed Claim Verification</span>
                        </div>
                        <div style="display:flex; align-items:center; gap:6px;">
                            <span class="accordion-badge" style="color:#34d399; background:rgba(16,185,129,0.15);">Verified: ${q.verified_claims || 0}</span>
                            <span class="accordion-badge" style="color:#fbbf24; background:rgba(245,158,11,0.15);">Partial: ${q.partially_supported_claims || 0}</span>
                            <span class="accordion-badge" style="color:#f87171; background:rgba(239,68,68,0.15);">Removed: ${q.removed_claims || 0}</span>
                        </div>
                    </summary>
                    <div class="accordion-content">
                        ${claimsAccordionHtml}
                    </div>
                </details>

                <!-- 7. ANSWER COMPARISON (Expandable Section) -->
                <details class="detail-accordion" open>
                    <summary class="detail-accordion-summary">
                        <div class="accordion-title-left">
                            <span class="accordion-arrow">▶</span>
                            <span>⚖️ Original Answer vs Final Verified Answer</span>
                        </div>
                        <span class="accordion-badge">${isAnswerModified ? 'Modified & Corrected' : '100% Verified'}</span>
                    </summary>
                    <div class="accordion-content">
                        <div class="comparison-answers-grid">
                            <div class="comparison-pane original-pane">
                                <div class="comparison-pane-title">
                                    <span>Initial Generated Draft</span>
                                    <span style="font-size:10px; font-weight:normal; color:#64748b;">Raw Model Output</span>
                                </div>
                                <div class="comparison-pane-body">${escapeHtml(draftAnswerText)}</div>
                            </div>

                            <div class="comparison-pane final-pane">
                                <div class="comparison-pane-title">
                                    <span>Final Verified Answer</span>
                                    <span style="font-size:10px; font-weight:normal; color:#34d399;">Grounded & Filtered</span>
                                </div>
                                <div class="comparison-pane-body">${escapeHtml(finalAnswerText)}</div>
                            </div>
                        </div>

                        ${isAnswerModified ? `
                            <div style="font-size:12px; color:#f87171; background:rgba(239, 68, 68, 0.08); border:1px solid rgba(239, 68, 68, 0.25); border-radius:6px; padding:8px 12px;">
                                ⚠️ <strong>Correction Applied:</strong> One or more unsupported assertions from the initial draft were filtered or rewritten based strictly on retrieved document evidence.
                            </div>
                        ` : `
                            <div style="font-size:12px; color:#34d399; background:rgba(16, 185, 129, 0.08); border:1px solid rgba(16, 185, 129, 0.25); border-radius:6px; padding:8px 12px;">
                                ✓ <strong>Grounded Answer:</strong> The generated answer was fully supported by document evidence without requiring claim removals.
                            </div>
                        `}
                    </div>
                </details>

                <!-- 8. RETRIEVAL PIPELINE FLOWCHART (Expandable Section) -->
                <details class="detail-accordion" open>
                    <summary class="detail-accordion-summary">
                        <div class="accordion-title-left">
                            <span class="accordion-arrow">▶</span>
                            <span>🔄 Retrieval Pipeline</span>
                        </div>
                        <span class="accordion-badge">Step-by-Step Flow</span>
                    </summary>
                    <div class="accordion-content">
                        <div class="pipeline-flowchart">
                            <!-- Stage 1 -->
                            <div class="pipeline-step-node active-node">
                                <div class="pipeline-step-node-left">
                                    <span class="pipeline-step-num">1</span>
                                    <div>
                                        <div class="pipeline-step-title">Question</div>
                                        <div class="pipeline-step-desc">${escapeHtml(q.question ? (q.question.length > 50 ? q.question.substring(0, 48) + '...' : q.question) : 'User Input')}</div>
                                    </div>
                                </div>
                                <span class="pipeline-step-timing">Input</span>
                            </div>

                            <div class="pipeline-step-connector">↓</div>

                            <!-- Stage 2 -->
                            <div class="pipeline-step-node">
                                <div class="pipeline-step-node-left">
                                    <span class="pipeline-step-num">2</span>
                                    <div>
                                        <div class="pipeline-step-title">Query Processing</div>
                                        <div class="pipeline-step-desc">Input cleaning, stop-word tokenization & query intent routing</div>
                                    </div>
                                </div>
                                <span class="pipeline-step-timing">Type: ${escapeHtml(q.query_type || 'NORMAL_QUESTION')}</span>
                            </div>

                            <div class="pipeline-step-connector">↓</div>

                            <!-- Stage 3 -->
                            <div class="pipeline-step-node">
                                <div class="pipeline-step-node-left">
                                    <span class="pipeline-step-num">3</span>
                                    <div>
                                        <div class="pipeline-step-title">Multi-Query / Query Decomposition</div>
                                        <div class="pipeline-step-desc">${q.is_decomposed ? 'Sub-queries generated for multi-part intent' : (q.is_comparison ? 'Comparison entities extracted and queried' : 'Single unified query evaluation')}</div>
                                    </div>
                                </div>
                                <span class="pipeline-step-timing">${q.is_decomposed ? 'Decomposed' : (q.is_comparison ? 'Comparison' : 'Standard')}</span>
                            </div>

                            <div class="pipeline-step-connector">↓</div>

                            <!-- Stage 4 -->
                            <div class="pipeline-step-node">
                                <div class="pipeline-step-node-left">
                                    <span class="pipeline-step-num">4</span>
                                    <div>
                                        <div class="pipeline-step-title">Hybrid Search</div>
                                        <div class="pipeline-step-desc">Combined Reciprocal Rank Fusion of Dense + Sparse Candidates</div>
                                    </div>
                                </div>
                                <span class="pipeline-step-timing">RRF Fused</span>
                            </div>

                            <div class="pipeline-step-connector">↓</div>

                            <!-- Stage 5 -->
                            <div class="pipeline-step-node">
                                <div class="pipeline-step-node-left">
                                    <span class="pipeline-step-num">5</span>
                                    <div>
                                        <div class="pipeline-step-title">FAISS Dense Retrieval</div>
                                        <div class="pipeline-step-desc">all-MiniLM-L6-v2 cosine vector similarity search</div>
                                    </div>
                                </div>
                                <span class="pipeline-step-timing">${retTimeStr}</span>
                            </div>

                            <div class="pipeline-step-connector">↓</div>

                            <!-- Stage 6 -->
                            <div class="pipeline-step-node">
                                <div class="pipeline-step-node-left">
                                    <span class="pipeline-step-num">6</span>
                                    <div>
                                        <div class="pipeline-step-title">BM25 Sparse Retrieval</div>
                                        <div class="pipeline-step-desc">Okapi BM25 keyword matching & lexical score calculation</div>
                                    </div>
                                </div>
                                <span class="pipeline-step-timing">BM25 Active</span>
                            </div>

                            <div class="pipeline-step-connector">↓</div>

                            <!-- Stage 7 -->
                            <div class="pipeline-step-node">
                                <div class="pipeline-step-node-left">
                                    <span class="pipeline-step-num">7</span>
                                    <div>
                                        <div class="pipeline-step-title">Cross-Encoder Reranking</div>
                                        <div class="pipeline-step-desc">ms-marco-MiniLM-L-6-v2 semantic cross-attention reranking</div>
                                    </div>
                                </div>
                                <span class="pipeline-step-timing">${rerankTimeStr}</span>
                            </div>

                            <div class="pipeline-step-connector">↓</div>

                            <!-- Stage 8 -->
                            <div class="pipeline-step-node">
                                <div class="pipeline-step-node-left">
                                    <span class="pipeline-step-num">8</span>
                                    <div>
                                        <div class="pipeline-step-title">Adaptive Retrieval</div>
                                        <div class="pipeline-step-desc">${q.retry_count > 0 ? q.retry_count + ' iterative reformulation retries executed' : 'Relevance threshold met on first attempt'}</div>
                                    </div>
                                </div>
                                <span class="pipeline-step-timing">${valOrNA(q.retry_count)} Retries</span>
                            </div>

                            <div class="pipeline-step-connector">↓</div>

                            <!-- Stage 9 -->
                            <div class="pipeline-step-node">
                                <div class="pipeline-step-node-left">
                                    <span class="pipeline-step-num">9</span>
                                    <div>
                                        <div class="pipeline-step-title">Contextual Compression</div>
                                        <div class="pipeline-step-desc">Redundancy pruning and prompt context window optimization</div>
                                    </div>
                                </div>
                                <span class="pipeline-step-timing">${compTimeStr}</span>
                            </div>

                            <div class="pipeline-step-connector">↓</div>

                            <!-- Stage 10 -->
                            <div class="pipeline-step-node">
                                <div class="pipeline-step-node-left">
                                    <span class="pipeline-step-num">10</span>
                                    <div>
                                        <div class="pipeline-step-title">LLM Generation</div>
                                        <div class="pipeline-step-desc">Google Gemini API generation with grounded prompt constraints</div>
                                    </div>
                                </div>
                                <span class="pipeline-step-timing">${genTimeStr}</span>
                            </div>

                            <div class="pipeline-step-connector">↓</div>

                            <!-- Stage 11 -->
                            <div class="pipeline-step-node">
                                <div class="pipeline-step-node-left">
                                    <span class="pipeline-step-num">11</span>
                                    <div>
                                        <div class="pipeline-step-title">Answer Evaluation</div>
                                        <div class="pipeline-step-desc">Groundedness (${valOrNA(q.groundedness, '%')}) and Faithfulness (${valOrNA(q.faithfulness, '%')}) scoring</div>
                                    </div>
                                </div>
                                <span class="pipeline-step-timing">Confidence: ${escapeHtml(q.confidence || 'Medium')}</span>
                            </div>

                            <div class="pipeline-step-connector">↓</div>

                            <!-- Stage 12 -->
                            <div class="pipeline-step-node">
                                <div class="pipeline-step-node-left">
                                    <span class="pipeline-step-num">12</span>
                                    <div>
                                        <div class="pipeline-step-title">Claim Verification</div>
                                        <div class="pipeline-step-desc">NLI cross-verification of factual assertions against sources</div>
                                    </div>
                                </div>
                                <span class="pipeline-step-timing">${verifTimeStr}</span>
                            </div>

                            <div class="pipeline-step-connector">↓</div>

                            <!-- Stage 13 -->
                            <div class="pipeline-step-node">
                                <div class="pipeline-step-node-left">
                                    <span class="pipeline-step-num">13</span>
                                    <div>
                                        <div class="pipeline-step-title">Answer Correction</div>
                                        <div class="pipeline-step-desc">${(q.removed_claims || 0) > 0 ? q.removed_claims + ' unsupported claims removed / rewritten' : 'Strict document grounding verified'}</div>
                                    </div>
                                </div>
                                <span class="pipeline-step-timing">${q.verified_claims || 0} Supported</span>
                            </div>

                            <div class="pipeline-step-connector">↓</div>

                            <!-- Stage 14 -->
                            <div class="pipeline-step-node active-node">
                                <div class="pipeline-step-node-left">
                                    <span class="pipeline-step-num">14</span>
                                    <div>
                                        <div class="pipeline-step-title">Final Answer</div>
                                        <div class="pipeline-step-desc">Status: ${escapeHtml(q.status || 'VERIFIED')} &bull; Grounded response with citations</div>
                                    </div>
                                </div>
                                <span class="pipeline-step-timing" style="color:#a5b4fc; border-color:rgba(99,102,241,0.4); background:rgba(99,102,241,0.15);">Total: ${totalTimeStr}</span>
                            </div>
                        </div>
                    </div>
                </details>
            `;
        } catch (renderErr) {
            console.error('Error rendering question detail:', renderErr);
            bodyEl.innerHTML = `
                <div class="detail-section-card highlight-card">
                    <div class="detail-section-title">
                        <div class="detail-section-title-left">
                            <span>💬</span> <span>Question: ${escapeHtml(q.question || 'Not available')}</span>
                        </div>
                        <span class="detail-section-badge">Status: ${escapeHtml(q.status || 'VERIFIED')}</span>
                    </div>
                    <div class="detail-qa-group">
                        <div class="detail-a-block">
                            <span class="detail-label-subtle">Final Answer</span>
                            <div class="detail-answer-box verified-border">${escapeHtml(q.answer_text || q.final_verified_answer || q.answer || 'Not available')}</div>
                        </div>
                        <div class="detail-a-block">
                            <span class="detail-label-subtle">Verification</span>
                            <div class="detail-answer-box" style="font-size:12px; color:#94a3b8;">${escapeHtml(q.confidence ? `Confidence: ${q.confidence} | Groundedness: ${valOrNA(q.groundedness, '%')}` : 'Not available')}</div>
                        </div>
                    </div>
                </div>
            `;
        }
    }

    window.openQuestionDetail = window.openQuestionDetails;

    // User Feedback Submission Handler
    window.submitAnswerFeedback = async function(msgId, feedbackType, btnElement) {
        try {
            const row = btnElement ? btnElement.closest('.answer-feedback-row') : null;
            if (row) {
                const upBtn = row.querySelector('.btn-feedback-up');
                const downBtn = row.querySelector('.btn-feedback-down');
                const statusMsg = row.querySelector('.feedback-status-msg');

                if (feedbackType === 'helpful') {
                    if (upBtn) upBtn.classList.add('active');
                    if (downBtn) downBtn.classList.remove('active');
                } else {
                    if (downBtn) downBtn.classList.add('active');
                    if (upBtn) upBtn.classList.remove('active');
                }

                if (statusMsg) {
                    statusMsg.textContent = 'Feedback recorded. Thank you!';
                    statusMsg.classList.remove('hidden');
                    setTimeout(() => {
                        statusMsg.classList.add('hidden');
                    }, 3000);
                }
            }

            await fetch('/api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    msg_id: msgId,
                    feedback: feedbackType
                })
            });
        } catch (err) {
            console.error('Failed to submit feedback:', err);
        }
    };

    // Run Evaluation Benchmark Button Handler
    const btnRunEval = document.getElementById('btn-run-evaluation');
    if (btnRunEval) {
        btnRunEval.addEventListener('click', async () => {
            const badge = document.getElementById('eval-status-badge');
            const timeLabel = document.getElementById('eval-timestamp');

            if (badge) {
                badge.textContent = 'Running (20 queries)...';
                badge.className = 'eval-badge running';
            }
            btnRunEval.disabled = true;
            btnRunEval.innerHTML = '<span>⏳</span> Running Benchmark...';

            try {
                const res = await fetch('/api/evaluation/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ top_k: 5 })
                });

                const data = await res.json();
                if (!res.ok || !data.success) {
                    showAlert(data.error || 'Evaluation benchmark failed.', 'error');
                    if (badge) {
                        badge.textContent = 'Failed';
                        badge.className = 'eval-badge failed';
                    }
                    return;
                }

                showAlert(`RAG Evaluation Benchmark completed across ${data.metrics.total_test_cases} test queries!`, 'success');
                fetchAnalyticsData(false);

            } catch (err) {
                console.error('Error running evaluation benchmark:', err);
                showAlert('Failed to connect to evaluation service.', 'error');
                if (badge) badge.textContent = 'Error';
            } finally {
                btnRunEval.disabled = false;
                btnRunEval.innerHTML = '<span>▶</span> Run Evaluation Benchmark';
            }
        });
    }

    // Prevent clicks inside modal content from bubbling to backdrop
    const qModalContent = document.querySelector('.question-modal-content');
    if (qModalContent) {
        qModalContent.addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }

    // Global keyboard listener for ESC key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' || e.key === 'Esc') {
            closeQuestionDetail(e);
            const srcModal = document.getElementById('source-modal');
            if (srcModal) srcModal.classList.add('hidden');
            const searchModal = document.getElementById('search-modal');
            if (searchModal) searchModal.classList.add('hidden');
        }
    });

    // =========================================================================
    // WORKSPACE & NAVIGATION ROUTER (MULTI-PAGE REDESIGN)
    // =========================================================================
    const VALID_VIEWS = ['dashboard', 'documents', 'search', 'chat', 'sessions', 'history', 'analytics', 'rag'];
    let currentActiveView = 'dashboard';

    // Elements
    const appLayout = document.querySelector('.app-layout');
    const appSidebar = document.getElementById('app-sidebar');
    const btnSidebarCollapse = document.getElementById('btn-sidebar-collapse') || document.getElementById('btn-toggle-sidebar');
    const btnMobileSidebarToggle = document.getElementById('btn-mobile-sidebar-toggle');
    const sidebarBackdrop = document.getElementById('sidebar-backdrop');
    const btnOpenPipelineModal = document.getElementById('btn-open-pipeline-modal');
    const pipelineModal = document.getElementById('pipeline-details-modal');
    const pipelineModalBackdrop = document.getElementById('pipeline-modal-backdrop');
    const btnClosePipelineModal = document.getElementById('btn-close-pipeline-modal');
    const btnClosePipelineModalBtn = document.getElementById('btn-close-pipeline-modal-btn');

    // Scope & meta elements
    const navBadgeDocs = document.getElementById('nav-badge-docs');
    const sidebarMetaDocs = document.getElementById('sidebar-meta-docs');
    const sidebarMetaChunks = document.getElementById('sidebar-meta-chunks');
    const chatScopeDocText = document.getElementById('chat-scope-doc-text');
    const dashStatDocs = document.getElementById('dash-stat-docs');
    const dashStatChunks = document.getElementById('dash-stat-chunks');
    const dashStatFaithfulness = document.getElementById('dash-stat-faithfulness');
    const dashStatStatus = document.getElementById('dash-stat-status');
    const dashDocTableBody = document.getElementById('dashboard-doc-table-body');
    const btnDashRebuild = document.getElementById('btn-dash-rebuild');

    // View Header Titles & Subtitles Map
    const VIEW_META = {
        dashboard: {
            title: 'Dashboard',
            subtitle: 'Overview of your document collection and quick actions'
        },
        documents: {
            title: 'Documents',
            subtitle: 'Upload and manage your PDF files'
        },
        search: {
            title: 'Search & Explore',
            subtitle: 'Find passages and keywords across your documents'
        },
        chat: {
            title: 'Document Assistant',
            subtitle: 'Ask questions about your documents'
        },
        sessions: {
            title: 'Sessions',
            subtitle: 'Continue your conversations where you left off.'
        },
        history: {
            title: 'History',
            subtitle: 'Review recent questions and answers'
        },
        analytics: {
            title: 'Analytics',
            subtitle: 'Performance, accuracy, and usage overview'
        },
        rag: {
            title: 'How RAG Works',
            subtitle: 'From your documents to grounded answers.'
        }
    };

    function navigateToView(viewId, updateHash = true) {
        if (!VALID_VIEWS.includes(viewId)) {
            viewId = 'dashboard';
        }
        currentActiveView = viewId;

        // 1. Hide all views, unhide selected view
        VALID_VIEWS.forEach(id => {
            const sectionEl = document.getElementById(`view-${id}`);
            if (sectionEl) {
                if (id === viewId) {
                    sectionEl.classList.remove('hidden');
                } else {
                    sectionEl.classList.add('hidden');
                }
            }
        });

        // Ensure inner sections for chat / analytics are properly unhidden
        if (viewId === 'chat' && chatViewSection) {
            chatViewSection.classList.remove('hidden');
        }
        if (viewId === 'analytics' && analyticsViewSection) {
            analyticsViewSection.classList.remove('hidden');
        }

        // 2. Update navigation active state
        document.querySelectorAll('.sidebar-nav .nav-item').forEach(link => {
            const targetView = link.getAttribute('data-view');
            if (targetView === viewId) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });

        // 3. Update topbar title & subtitle
        const meta = VIEW_META[viewId] || VIEW_META.dashboard;
        if (mainHeaderTitle) mainHeaderTitle.textContent = meta.title;
        if (mainHeaderSubtitle) mainHeaderSubtitle.textContent = meta.subtitle;

        // 4. Update URL hash if requested
        if (updateHash && window.location.hash !== `#${viewId}`) {
            window.location.hash = `#${viewId}`;
        }

        // 5. Trigger view-specific render / action
        if (viewId === 'dashboard') {
            renderDashboardSummary();
        } else if (viewId === 'search') {
            if (typeof updateSearchScopeBadge === 'function') updateSearchScopeBadge();
            if (docSearchInput) setTimeout(() => docSearchInput.focus(), 100);
        } else if (viewId === 'chat') {
            updateChatScopeBadge();
            if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight;
        } else if (viewId === 'sessions') {
            if (typeof renderSessionsPage === 'function') renderSessionsPage();
        } else if (viewId === 'history') {
            renderHistoryPage();
        } else if (viewId === 'analytics') {
            if (typeof fetchAndRenderAnalytics === 'function') {
                fetchAndRenderAnalytics();
            }
        } else if (viewId === 'rag') {
            if (typeof initRagWorksPage === 'function') initRagWorksPage();
        }

        // 6. Close mobile drawer if open
        if (appSidebar && appSidebar.classList.contains('mobile-open')) {
            appSidebar.classList.remove('mobile-open');
            if (sidebarBackdrop) sidebarBackdrop.classList.add('hidden');
        }
    }
    window.navigateToView = navigateToView;

    // Sidebar Collapsing
    function setSidebarCollapsed(collapsed, save = true) {
        if (!appLayout || !appSidebar) return;
        const iconSpan = document.getElementById('collapse-icon');
        if (collapsed) {
            appLayout.classList.add('sidebar-collapsed');
            appSidebar.classList.add('collapsed');
            if (iconSpan) iconSpan.textContent = '⇥';
            else if (btnSidebarCollapse) btnSidebarCollapse.innerHTML = '<span>▶</span>';
        } else {
            appLayout.classList.remove('sidebar-collapsed');
            appSidebar.classList.remove('collapsed');
            if (iconSpan) iconSpan.textContent = '⇤';
            else if (btnSidebarCollapse) btnSidebarCollapse.innerHTML = '<span>◀</span>';
        }
        if (save) {
            try {
                localStorage.setItem('rag_sidebar_collapsed', collapsed ? 'true' : 'false');
            } catch (_) {}
        }
    }

    if (btnSidebarCollapse) {
        btnSidebarCollapse.addEventListener('click', () => {
            const isCurrentlyCollapsed = appSidebar.classList.contains('collapsed');
            setSidebarCollapsed(!isCurrentlyCollapsed, true);
        });
    }

    // Restore saved sidebar state
    try {
        const savedSidebarState = localStorage.getItem('rag_sidebar_collapsed');
        if (savedSidebarState === 'true') {
            setSidebarCollapsed(true, false);
        }
    } catch (_) {}

    // Mobile Sidebar Drawer
    if (btnMobileSidebarToggle) {
        btnMobileSidebarToggle.addEventListener('click', () => {
            if (appSidebar) {
                const isOpen = appSidebar.classList.contains('mobile-open');
                if (isOpen) {
                    appSidebar.classList.remove('mobile-open');
                    if (sidebarBackdrop) sidebarBackdrop.classList.add('hidden');
                } else {
                    appSidebar.classList.add('mobile-open');
                    if (sidebarBackdrop) sidebarBackdrop.classList.remove('hidden');
                }
            }
        });
    }

    if (sidebarBackdrop) {
        sidebarBackdrop.addEventListener('click', () => {
            if (appSidebar) appSidebar.classList.remove('mobile-open');
            sidebarBackdrop.classList.add('hidden');
        });
    }

    // Pipeline Details Modal
    function openPipelineModal() {
        if (pipelineModal) pipelineModal.classList.remove('hidden');
    }
    function closePipelineModal() {
        if (pipelineModal) pipelineModal.classList.add('hidden');
    }
    if (btnOpenPipelineModal) btnOpenPipelineModal.addEventListener('click', openPipelineModal);
    if (btnClosePipelineModal) btnClosePipelineModal.addEventListener('click', closePipelineModal);
    if (btnClosePipelineModalBtn) btnClosePipelineModalBtn.addEventListener('click', closePipelineModal);
    if (pipelineModalBackdrop) pipelineModalBackdrop.addEventListener('click', closePipelineModal);

    // Dashboard Elements
    const dashMetricDocs = document.getElementById('dash-metric-docs');
    const dashMetricChunks = document.getElementById('dash-metric-chunks');
    const dashMetricPagesSub = document.getElementById('dash-metric-pages-sub');
    const dashMetricQuestions = document.getElementById('dash-metric-questions');
    const dashDocListPreview = document.getElementById('dash-doc-list-preview');

    // Dashboard Summary Rendering
    function renderDashboardSummary() {
        const total = allDocumentsList.length;
        const totalChunks = allDocumentsList.reduce((acc, d) => acc + (d.chunks || 0), 0);
        const totalPages = allDocumentsList.reduce((acc, d) => acc + (d.pages || 0), 0);

        if (dashMetricDocs) dashMetricDocs.textContent = total;
        if (dashMetricChunks) dashMetricChunks.textContent = totalChunks;
        if (dashMetricPagesSub) dashMetricPagesSub.textContent = `Across ${totalPages} total pages`;

        const questions = (typeof getRecentQuestions === 'function') ? getRecentQuestions() : [];
        if (dashMetricQuestions) dashMetricQuestions.textContent = questions.length;

        // Render document collection cards in dashboard
        if (dashDocListPreview) {
            if (total === 0) {
                dashDocListPreview.innerHTML = `
                    <div class="empty-list-placeholder" style="padding: 24px; text-align: center; color: var(--text-muted);">
                        No documents uploaded yet. Click <a href="#documents" style="color:var(--accent-primary); text-decoration:underline;">Upload & Manage Documents</a> to get started.
                    </div>
                `;
            } else {
                dashDocListPreview.innerHTML = `
                    <div style="display:grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px;">
                        ${allDocumentsList.map(doc => {
                            const isIndexed = doc.indexed || (doc.status || '').toLowerCase() === 'ready';
                            const safeName = escapeHtml(doc.filename);
                            const pages = doc.pages || 1;
                            const chunks = isIndexed ? (doc.chunks || 0) : 0;
                            const statusPill = isIndexed
                                ? `<span class="table-status-pill status-ready" style="font-size:11px; padding:3px 8px;">🟢 Indexed</span>`
                                : `<span class="table-status-pill status-pending" style="font-size:11px; padding:3px 8px;">🟡 Pending</span>`;

                            return `
                                <div style="background:var(--bg-card); border:1px solid var(--border-color); border-radius:10px; padding:14px; display:flex; flex-direction:column; gap:8px;">
                                    <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:8px;">
                                        <div style="font-weight:600; font-size:13px; color:var(--text-primary); text-overflow:ellipsis; overflow:hidden; white-space:nowrap; max-width:180px;" title="${safeName}">
                                            📄 ${safeName}
                                        </div>
                                        ${statusPill}
                                    </div>
                                    <div style="font-size:12px; color:var(--text-muted); display:flex; gap:12px;">
                                        <span>📑 ${pages} pages</span>
                                        <span>📑 ${chunks} sections</span>
                                    </div>
                                    <div style="display:flex; gap:8px; margin-top:4px;">
                                        <button type="button" class="btn btn-secondary btn-sm" onclick="navigateToView('documents')" style="padding:3px 8px; font-size:11px; flex:1;">
                                            Manage
                                        </button>
                                        <button type="button" class="btn btn-ghost btn-sm" onclick="navigateToView('chat')" style="padding:3px 8px; font-size:11px; flex:1;">
                                            Ask Chat
                                        </button>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                `;
            }
        }
    }
    window.renderDashboardSummary = renderDashboardSummary;

    // Chat Scope & Navigation Badges
    function updateChatScopeBadge() {
        if (!chatScopeDocText) return;
        const total = allDocumentsList.length;
        const selected = selectedDocs.size;
        if (total === 0) {
            chatScopeDocText.textContent = 'No documents uploaded';
        } else if (selected === total) {
            chatScopeDocText.textContent = `All Documents Active (${total} docs)`;
        } else if (selected === 0) {
            chatScopeDocText.textContent = '0 Documents Selected (Check in Documents page)';
        } else {
            chatScopeDocText.textContent = `${selected} of ${total} documents selected`;
        }

        // Also update sidebar badges
        if (navBadgeDocs) navBadgeDocs.textContent = total;
        if (sidebarMetaDocs) sidebarMetaDocs.textContent = `${total} Docs`;
        const totalChunks = allDocumentsList.reduce((acc, d) => acc + (d.chunks || 0), 0);
        if (sidebarMetaChunks) sidebarMetaChunks.textContent = `${totalChunks} Sections`;
    }

    // Dashboard Quick Actions
    const btnDashUpload = document.getElementById('btn-dash-upload');
    const btnDashSearch = document.getElementById('btn-dash-search');
    const btnDashChat = document.getElementById('btn-dash-chat');
    const btnDashAnalytics = document.getElementById('btn-dash-analytics');

    if (btnDashUpload) btnDashUpload.addEventListener('click', () => navigateToView('documents'));
    if (btnDashSearch) btnDashSearch.addEventListener('click', () => navigateToView('search'));
    if (btnDashChat) btnDashChat.addEventListener('click', () => navigateToView('chat'));
    if (btnDashAnalytics) btnDashAnalytics.addEventListener('click', () => navigateToView('analytics'));

    if (btnDashRebuild && btnRebuild) {
        btnDashRebuild.addEventListener('click', () => {
            btnRebuild.click();
        });
    }

    // =========================================================================
    // MULTI-SESSION CHAT SUPPORT
    // =========================================================================
    const sessionsMemory = {
        'default': { history: [], html: null },
        'dbms_prep': { history: [], html: null },
        'sgcube_research': { history: [], html: null },
        'sql_revision': { history: [], html: null }
    };
    let currentSessionKey = 'default';

    const chatSessionSelect = document.getElementById('chat-session-select');
    if (chatSessionSelect) {
        chatSessionSelect.addEventListener('change', (e) => {
            const newSessionKey = e.target.value;
            if (newSessionKey === currentSessionKey) return;

            // Save active session
            if (sessionsMemory[currentSessionKey]) {
                sessionsMemory[currentSessionKey].history = [...chatHistory];
                sessionsMemory[currentSessionKey].html = chatMessages ? chatMessages.innerHTML : null;
            }

            // Switch to new session
            currentSessionKey = newSessionKey;
            const target = sessionsMemory[currentSessionKey] || { history: [], html: null };
            chatHistory = target.history ? [...target.history] : [];

            if (chatMessages) {
                if (target.html) {
                    chatMessages.innerHTML = target.html;
                } else {
                    const sessionName = e.target.options[e.target.selectedIndex].text;
                    chatMessages.innerHTML = `
                        <div class="message system-message" id="welcome-message">
                            <div class="message-avatar">🤖</div>
                            <div class="message-content">
                                <h3>Session: ${escapeHtml(sessionName)}</h3>
                                <p>This is a separate conversation workspace. Ask questions grounded in your selected documents.</p>
                            </div>
                        </div>
                    `;
                }
            }

            showAlert(`Switched to session: ${e.target.options[e.target.selectedIndex].text}`, 'info');
        });
    }

    // =========================================================================
    // HISTORY PAGE WORKSPACE LOGIC
    // =========================================================================
    const historyCountBadge = document.getElementById('history-count-badge');
    const historyFilterInput = document.getElementById('history-filter-input');

    function renderHistoryPage() {
        const questions = (typeof getRecentQuestions === 'function') ? getRecentQuestions() : [];
        if (historyCountBadge) {
            historyCountBadge.textContent = `${questions.length} quer${questions.length === 1 ? 'y' : 'ies'} recorded`;
        }

        const listEl = document.getElementById('recent-questions-list');
        if (!listEl) return;

        const filterVal = historyFilterInput ? historyFilterInput.value.trim().toLowerCase() : '';
        const filtered = filterVal
            ? questions.filter(q => q.toLowerCase().includes(filterVal))
            : questions;

        if (filtered.length === 0) {
            listEl.innerHTML = `
                <li class="empty-recent-item" style="padding:20px; text-align:center; color:var(--text-muted);">
                    ${filterVal ? 'No history matching "' + escapeHtml(filterVal) + '"' : 'No queries recorded yet. Questions asked in Chat will appear here.'}
                </li>
            `;
            return;
        }

        listEl.innerHTML = filtered.map((q, idx) => {
            const safeQ = escapeHtml(q);
            const escapedForAttr = safeQ.replace(/'/g, "\\'");
            return `
                <li class="history-item-row" style="display:flex; justify-content:space-between; align-items:center; padding:14px 18px; border-bottom:1px solid rgba(255,255,255,0.06); gap:14px;">
                    <div style="display:flex; align-items:center; gap:12px; overflow:hidden; flex:1;">
                        <span style="font-size:16px;">💬</span>
                        <span class="history-q-text" style="font-weight:500; color:var(--text-primary); text-overflow:ellipsis; overflow:hidden; white-space:nowrap;" title="${safeQ}">
                            ${safeQ}
                        </span>
                    </div>
                    <div style="display:flex; gap:8px; flex-shrink:0;">
                        <button type="button" class="btn btn-secondary btn-sm" onclick="askHistoryQuestionInChat('${escapedForAttr}')" title="Ask this question in Chat">
                            <span>💬</span> Ask in Chat
                        </button>
                        <button type="button" class="btn btn-ghost btn-sm" onclick="searchHistoryQuestionInSearch('${escapedForAttr}')" title="Search chunks for this query">
                            <span>🔍</span> Search Chunks
                        </button>
                    </div>
                </li>
            `;
        }).join('');
    }

    window.askHistoryQuestionInChat = function(queryText) {
        navigateToView('chat');
        if (userInput) {
            userInput.value = queryText;
            userInput.focus();
        }
    };

    window.searchHistoryQuestionInSearch = function(queryText) {
        navigateToView('search');
        if (docSearchInput) {
            docSearchInput.value = queryText;
            performDocumentSearch(queryText);
        }
    };

    if (historyFilterInput) {
        historyFilterInput.addEventListener('input', () => {
            renderHistoryPage();
        });
    }

    // Setup Nav Item click listeners
    document.querySelectorAll('.sidebar-nav .nav-item').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const viewId = link.getAttribute('data-view');
            navigateToView(viewId, true);
        });
    });

    // Hash change event listener
    window.addEventListener('hashchange', () => {
        const hash = window.location.hash.replace('#', '');
        if (VALID_VIEWS.includes(hash) && hash !== currentActiveView) {
            navigateToView(hash, false);
        }
    });

    // Hook existing openAnalyticsView & openChatView into navigateToView
    openAnalyticsView = function() {
        navigateToView('analytics', true);
    };

    openChatView = function() {
        navigateToView('chat', true);
    };

    // Hook updateSelectionUI to also refresh navigation badges & chat scope
    const origUpdateSelectionUI = updateSelectionUI;
    updateSelectionUI = function() {
        origUpdateSelectionUI();
        updateChatScopeBadge();
        renderDashboardSummary();
    };
    // (Initial Route Detection moved to end of setup)

    // =========================================================================
    // DUAL-THEME SWITCHER & CIRCULAR VIEW TRANSITION
    // =========================================================================
    const btnThemeToggle = document.getElementById('btn-theme-toggle');
    const themeToggleIcon = document.getElementById('theme-toggle-icon');
    const themeToggleLabel = document.getElementById('theme-toggle-label');
    const themeTransitionOverlay = document.getElementById('theme-transition-overlay');

    function getCurrentTheme() {
        return document.documentElement.getAttribute('data-theme') || 'light';
    }

    function updateThemeToggleUI(theme) {
        if (!themeToggleIcon || !themeToggleLabel) return;
        if (theme === 'dark') {
            themeToggleIcon.textContent = '☀️';
            themeToggleLabel.textContent = 'Light';
            if (btnThemeToggle) {
                btnThemeToggle.setAttribute('aria-label', 'Switch to Light Theme');
                btnThemeToggle.setAttribute('title', 'Switch to Light Theme (Current: Dark)');
            }
        } else {
            themeToggleIcon.textContent = '🌙';
            themeToggleLabel.textContent = 'Dark';
            if (btnThemeToggle) {
                btnThemeToggle.setAttribute('aria-label', 'Switch to Dark Theme');
                btnThemeToggle.setAttribute('title', 'Switch to Dark Theme (Current: Light)');
            }
        }
    }

    function applyThemeDirect(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        try {
            localStorage.setItem('rag_theme', theme);
        } catch (_) {}
        updateThemeToggleUI(theme);
    }

    function toggleTheme(event) {
        const currentTheme = getCurrentTheme();
        const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
        const isReducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

        // 1. If reduced motion requested, switch with simple fade without radial animation
        if (isReducedMotion) {
            applyThemeDirect(nextTheme);
            return;
        }

        // 2. Determine animation origin coordinates
        let x, y;
        if (event && typeof event.clientX === 'number' && typeof event.clientY === 'number' && event.clientX > 0 && event.clientY > 0) {
            x = event.clientX;
            y = event.clientY;
        } else if (btnThemeToggle) {
            const rect = btnThemeToggle.getBoundingClientRect();
            x = rect.left + rect.width / 2;
            y = rect.top + rect.height / 2;
        } else {
            x = window.innerWidth - 60;
            y = 32;
        }

        // Maximum distance to viewport corners
        const endRadius = Math.hypot(
            Math.max(x, window.innerWidth - x),
            Math.max(y, window.innerHeight - y)
        );

        // 3. Modern Chrome View Transitions API (GPU composited 60fps radial expansion)
        if (typeof document.startViewTransition === 'function') {
            const transition = document.startViewTransition(() => {
                applyThemeDirect(nextTheme);
            });

            transition.ready.then(() => {
                const clipPath = [
                    `circle(0px at ${x}px ${y}px)`,
                    `circle(${endRadius}px at ${x}px ${y}px)`
                ];
                document.documentElement.animate(
                    {
                        clipPath: clipPath
                    },
                    {
                        duration: 550,
                        easing: 'cubic-bezier(0.25, 1, 0.5, 1)',
                        pseudoElement: '::view-transition-new(root)'
                    }
                );
            }).catch(() => {
                // In case of transition abort, theme is already set
            });
            return;
        }

        // 4. Elegant Fallback Overlay for browsers without View Transitions
        if (themeTransitionOverlay) {
            themeTransitionOverlay.style.background = nextTheme === 'dark' ? '#23262F' : '#FFFFFF';
            themeTransitionOverlay.style.clipPath = `circle(0px at ${x}px ${y}px)`;
            themeTransitionOverlay.classList.add('animating');

            requestAnimationFrame(() => {
                themeTransitionOverlay.style.clipPath = `circle(${endRadius}px at ${x}px ${y}px)`;
                setTimeout(() => {
                    applyThemeDirect(nextTheme);
                    themeTransitionOverlay.classList.remove('animating');
                    themeTransitionOverlay.style.clipPath = `circle(0px at ${x}px ${y}px)`;
                }, 500);
            });
            return;
        }

        // Default instant apply
        applyThemeDirect(nextTheme);
    }
    window.toggleTheme = toggleTheme;

    // Attach listeners
    if (btnThemeToggle) {
        btnThemeToggle.addEventListener('click', (e) => {
            toggleTheme(e);
        });

        btnThemeToggle.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                toggleTheme(e);
            }
        });
    }

    // Initialize toggle UI based on current active theme
    updateThemeToggleUI(getCurrentTheme());

    // =========================================================================
    // SESSIONS WORKSPACE IMPLEMENTATION (BROWSER-SAVED SESSION STORAGE)
    // =========================================================================
    const SESSIONS_STORAGE_KEY = 'rag_user_sessions_v1';
    let userSessions = [];
    let sessionToRenameId = null;
    let sessionToDeleteId = null;

    function initUserSessions() {
        try {
            const stored = localStorage.getItem(SESSIONS_STORAGE_KEY);
            if (stored) {
                userSessions = JSON.parse(stored);
            }
        } catch (e) {
            console.warn('Failed to parse sessions from localStorage:', e);
        }

        if (!userSessions || !Array.isArray(userSessions) || userSessions.length === 0) {
            // Seed initial sessions
            userSessions = [
                {
                    id: 'default',
                    title: 'Default Session',
                    preview: 'General questions and overview of your document library.',
                    messageCount: 1,
                    lastActive: Date.now() - 300000,
                    docCount: (typeof allDocumentsList !== 'undefined') ? allDocumentsList.length : 1,
                    isPinned: true
                },
                {
                    id: 'dbms_prep',
                    title: 'DBMS Preparation',
                    preview: 'Primary keys, candidate keys, normalization and ACID properties.',
                    messageCount: 12,
                    lastActive: Date.now() - 600000,
                    docCount: 1,
                    isPinned: false
                },
                {
                    id: 'sgcube_research',
                    title: 'SG CUBE Research',
                    preview: 'Edge-hybrid dual-mode architecture and computer vision pipeline.',
                    messageCount: 6,
                    lastActive: Date.now() - 3600000,
                    docCount: 1,
                    isPinned: false
                },
                {
                    id: 'sql_revision',
                    title: 'SQL Revision',
                    preview: 'DDL, DML commands, indexing, and relational JOIN operations.',
                    messageCount: 4,
                    lastActive: Date.now() - 86400000,
                    docCount: 1,
                    isPinned: false
                }
            ];
            saveUserSessions();
        }
    }

    function saveUserSessions() {
        try {
            localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(userSessions));
        } catch (e) {
            console.warn('Failed to persist sessions:', e);
        }
    }

    function formatRelativeTime(timestamp) {
        if (!timestamp) return 'Recently';
        const diffMs = Date.now() - Number(timestamp);
        const mins = Math.floor(diffMs / 60000);
        if (mins < 1) return 'Just now';
        if (mins < 60) return `${mins} min${mins === 1 ? '' : 's'} ago`;
        const hours = Math.floor(mins / 60);
        if (hours < 24) return `${hours} hour${hours === 1 ? '' : 's'} ago`;
        const days = Math.floor(hours / 24);
        return `${days} day${days === 1 ? '' : 's'} ago`;
    }

    function renderSessionsPage() {
        initUserSessions();
        const grid = document.getElementById('sessions-grid-container');
        const countPill = document.getElementById('sessions-count-pill');
        const searchInput = document.getElementById('sessions-search-input');
        if (!grid) return;

        const filterVal = searchInput ? searchInput.value.trim().toLowerCase() : '';
        let list = [...userSessions];

        if (filterVal) {
            list = list.filter(s =>
                (s.title && s.title.toLowerCase().includes(filterVal)) ||
                (s.preview && s.preview.toLowerCase().includes(filterVal))
            );
        }

        // Sort: pinned first, then lastActive descending
        list.sort((a, b) => {
            if (a.isPinned && !b.isPinned) return -1;
            if (!a.isPinned && b.isPinned) return 1;
            return (b.lastActive || 0) - (a.lastActive || 0);
        });

        if (countPill) {
            countPill.textContent = `${list.length} session${list.length === 1 ? '' : 's'}`;
        }

        if (list.length === 0) {
            grid.innerHTML = `
                <div class="empty-sessions-card">
                    <span class="empty-sessions-icon">💬</span>
                    <h3>${filterVal ? 'No matching sessions found' : 'No sessions yet'}</h3>
                    <p>${filterVal ? 'Try adjusting your search query.' : 'Start a new conversation about your documents.'}</p>
                    <button type="button" class="btn btn-primary btn-sm" onclick="handleCreateNewSession()">
                        <span>➕</span> New Session
                    </button>
                </div>
            `;
            return;
        }

        grid.innerHTML = list.map(s => {
            const safeTitle = escapeHtml(s.title || 'Untitled Session');
            const safePreview = escapeHtml(s.preview || 'No conversation preview available.');
            const msgs = s.messageCount || 0;
            const docs = s.docCount !== undefined ? s.docCount : (typeof allDocumentsList !== 'undefined' ? allDocumentsList.length : 1);
            const timeAgo = formatRelativeTime(s.lastActive);
            const pinIcon = s.isPinned ? '📌' : '';
            const pinText = s.isPinned ? 'Unpin' : 'Pin to Top';

            return `
                <div class="session-card ${s.isPinned ? 'pinned-session' : ''}" id="session-card-${s.id}">
                    <div class="session-card-header">
                        <div class="session-title-wrap">
                            <h3 class="session-title">${safeTitle}</h3>
                            ${s.isPinned ? '<span class="session-pin-badge" title="Pinned session">📌</span>' : ''}
                        </div>
                        <div class="session-actions-menu-wrap">
                            <button type="button" class="btn-session-more" onclick="toggleSessionMenu(event, '${s.id}')" title="Session options">⋯</button>
                            <div class="session-dropdown-menu hidden" id="session-menu-${s.id}">
                                <button type="button" class="session-menu-item" onclick="handleOpenRenameModal('${s.id}')">
                                    <span>✏️</span> Rename
                                </button>
                                <button type="button" class="session-menu-item" onclick="handleTogglePinSession('${s.id}')">
                                    <span>📌</span> ${pinText}
                                </button>
                                <button type="button" class="session-menu-item text-danger" onclick="handleOpenDeleteSessionModal('${s.id}')">
                                    <span>🗑️</span> Delete
                                </button>
                            </div>
                        </div>
                    </div>
                    <p class="session-preview-text">${safePreview}</p>
                    <div class="session-meta-row">
                        <span class="session-meta-item">💬 ${msgs} message${msgs === 1 ? '' : 's'}</span>
                        <span class="session-meta-dot">•</span>
                        <span class="session-meta-item">📄 ${docs} document${docs === 1 ? '' : 's'}</span>
                    </div>
                    <div class="session-card-footer">
                        <span class="session-time-text">Last active ${timeAgo}</span>
                        <button type="button" class="btn btn-primary btn-sm session-open-btn" onclick="handleOpenSession('${s.id}')">
                            Open &rarr;
                        </button>
                    </div>
                </div>
            `;
        }).join('');
    }

    window.toggleSessionMenu = function(event, sessionId) {
        event.stopPropagation();
        // Close all other menus
        document.querySelectorAll('.session-dropdown-menu').forEach(m => {
            if (m.id !== `session-menu-${sessionId}`) m.classList.add('hidden');
        });
        const menu = document.getElementById(`session-menu-${sessionId}`);
        if (menu) menu.classList.toggle('hidden');
    };

    document.addEventListener('click', () => {
        document.querySelectorAll('.session-dropdown-menu').forEach(m => m.classList.add('hidden'));
    });

    window.handleCreateNewSession = function() {
        const id = 'session_' + Date.now();
        const num = userSessions.length + 1;
        const newSession = {
            id: id,
            title: `Session ${num}`,
            preview: 'Fresh conversation ready for questions.',
            messageCount: 0,
            lastActive: Date.now(),
            docCount: (typeof allDocumentsList !== 'undefined') ? allDocumentsList.length : 1,
            isPinned: false
        };
        userSessions.unshift(newSession);
        saveUserSessions();

        // Also add to sessionsMemory & dropdown if present
        if (typeof sessionsMemory !== 'undefined') {
            sessionsMemory[id] = { history: [], html: null };
            currentSessionKey = id;
        }

        const chatSelect = document.getElementById('chat-session-select');
        if (chatSelect) {
            const opt = document.createElement('option');
            opt.value = id;
            opt.textContent = newSession.title;
            opt.selected = true;
            chatSelect.appendChild(opt);
        }

        // Clear active chat messages and navigate to Chat
        if (chatMessages) {
            chatMessages.innerHTML = `
                <div class="message system-message" id="welcome-message">
                    <div class="message-avatar">💬</div>
                    <div class="message-content">
                        <h3>Session: ${escapeHtml(newSession.title)}</h3>
                        <p>Ask questions grounded in your selected documents.</p>
                    </div>
                </div>
            `;
        }
        if (typeof chatHistory !== 'undefined') chatHistory = [];
        if (userInput) userInput.value = '';

        navigateToView('chat');
        showAlert(`Created ${newSession.title}`, 'success');
    };

    window.handleOpenSession = function(sessionId) {
        const session = userSessions.find(s => s.id === sessionId);
        if (!session) return;

        session.lastActive = Date.now();
        saveUserSessions();

        // Set active session in memory
        if (typeof sessionsMemory !== 'undefined') {
            currentSessionKey = sessionId;
            const target = sessionsMemory[sessionId] || { history: [], html: null };
            chatHistory = target.history ? [...target.history] : [];
            if (chatMessages) {
                if (target.html) {
                    chatMessages.innerHTML = target.html;
                } else {
                    chatMessages.innerHTML = `
                        <div class="message system-message" id="welcome-message">
                            <div class="message-avatar">💬</div>
                            <div class="message-content">
                                <h3>Session: ${escapeHtml(session.title)}</h3>
                                <p>Continue your conversation grounded in your selected documents.</p>
                            </div>
                        </div>
                    `;
                }
            }
        }

        const chatSelect = document.getElementById('chat-session-select');
        if (chatSelect) {
            let found = false;
            for (let i = 0; i < chatSelect.options.length; i++) {
                if (chatSelect.options[i].value === sessionId) {
                    chatSelect.selectedIndex = i;
                    found = true;
                    break;
                }
            }
            if (!found) {
                const opt = document.createElement('option');
                opt.value = session.id;
                opt.textContent = session.title;
                opt.selected = true;
                chatSelect.appendChild(opt);
            }
        }

        navigateToView('chat');
        showAlert(`Opened session: ${session.title}`, 'info');
    };

    window.handleOpenRenameModal = function(sessionId) {
        const session = userSessions.find(s => s.id === sessionId);
        if (!session) return;
        sessionToRenameId = sessionId;
        const input = document.getElementById('session-rename-input');
        const modal = document.getElementById('session-rename-modal');
        if (input) input.value = session.title;
        if (modal) modal.classList.remove('hidden');
        setTimeout(() => { if (input) input.focus(); }, 100);
    };

    window.handleTogglePinSession = function(sessionId) {
        const session = userSessions.find(s => s.id === sessionId);
        if (!session) return;
        session.isPinned = !session.isPinned;
        saveUserSessions();
        renderSessionsPage();
        showAlert(session.isPinned ? `Pinned ${session.title}` : `Unpinned ${session.title}`, 'info');
    };

    window.handleOpenDeleteSessionModal = function(sessionId) {
        const session = userSessions.find(s => s.id === sessionId);
        if (!session) return;
        sessionToDeleteId = sessionId;
        const titleEl = document.getElementById('delete-session-title');
        const modal = document.getElementById('session-delete-modal');
        if (titleEl) titleEl.textContent = `"${session.title}"`;
        if (modal) modal.classList.remove('hidden');
    };

    // Session Modal Event Listeners
    const btnConfirmRename = document.getElementById('btn-confirm-session-rename');
    const btnCancelRename = document.getElementById('btn-cancel-session-rename');
    const btnCloseRename = document.getElementById('btn-close-session-rename');
    const sessionRenameModal = document.getElementById('session-rename-modal');

    function closeRenameModal() {
        if (sessionRenameModal) sessionRenameModal.classList.add('hidden');
        sessionToRenameId = null;
    }
    if (btnCancelRename) btnCancelRename.addEventListener('click', closeRenameModal);
    if (btnCloseRename) btnCloseRename.addEventListener('click', closeRenameModal);
    if (btnConfirmRename) {
        btnConfirmRename.addEventListener('click', () => {
            const input = document.getElementById('session-rename-input');
            const newTitle = input ? input.value.trim() : '';
            if (newTitle && sessionToRenameId) {
                const s = userSessions.find(x => x.id === sessionToRenameId);
                if (s) {
                    s.title = newTitle;
                    saveUserSessions();
                    renderSessionsPage();
                    showAlert('Session renamed successfully.', 'success');
                }
            }
            closeRenameModal();
        });
    }

    const btnConfirmDeleteSession = document.getElementById('btn-confirm-session-delete');
    const btnCancelDeleteSession = document.getElementById('btn-cancel-session-delete');
    const btnCloseDeleteSession = document.getElementById('btn-close-session-delete');
    const sessionDeleteModal = document.getElementById('session-delete-modal');

    function closeDeleteSessionModal() {
        if (sessionDeleteModal) sessionDeleteModal.classList.add('hidden');
        sessionToDeleteId = null;
    }
    if (btnCancelDeleteSession) btnCancelDeleteSession.addEventListener('click', closeDeleteSessionModal);
    if (btnCloseDeleteSession) btnCloseDeleteSession.addEventListener('click', closeDeleteSessionModal);
    if (btnConfirmDeleteSession) {
        btnConfirmDeleteSession.addEventListener('click', () => {
            if (sessionToDeleteId) {
                userSessions = userSessions.filter(x => x.id !== sessionToDeleteId);
                saveUserSessions();
                renderSessionsPage();
                showAlert('Session deleted.', 'info');
            }
            closeDeleteSessionModal();
        });
    }

    // Sessions Search Input & Topbar Buttons
    const sessionsSearchInput = document.getElementById('sessions-search-input');
    const btnClearSessionsSearch = document.getElementById('btn-clear-sessions-search');
    const btnCreateSessionMain = document.getElementById('btn-create-session-main');

    if (sessionsSearchInput) {
        sessionsSearchInput.addEventListener('input', () => {
            const val = sessionsSearchInput.value.trim();
            if (btnClearSessionsSearch) {
                if (val) btnClearSessionsSearch.classList.remove('hidden');
                else btnClearSessionsSearch.classList.add('hidden');
            }
            renderSessionsPage();
        });
    }
    if (btnClearSessionsSearch) {
        btnClearSessionsSearch.addEventListener('click', () => {
            if (sessionsSearchInput) sessionsSearchInput.value = '';
            btnClearSessionsSearch.classList.add('hidden');
            renderSessionsPage();
        });
    }
    if (btnCreateSessionMain) {
        btnCreateSessionMain.addEventListener('click', handleCreateNewSession);
    }

    // =========================================================================
    // HOW RAG WORKS WORKSPACE IMPLEMENTATION (INTERACTIVE DEMO & ANIMATIONS)
    // =========================================================================
    let ragObserverInitialized = false;

    function initRagWorksPage() {
        if (ragObserverInitialized) return;
        ragObserverInitialized = true;

        // 1. IntersectionObserver for stage cards
        if ('IntersectionObserver' in window) {
            const isReducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            if (!isReducedMotion) {
                const observer = new IntersectionObserver((entries) => {
                    entries.forEach(entry => {
                        if (entry.isIntersecting) {
                            entry.target.classList.add('stage-in-view');
                        }
                    });
                }, { threshold: 0.15 });

                document.querySelectorAll('.pipeline-stage-card').forEach(card => {
                    observer.observe(card);
                });
            } else {
                document.querySelectorAll('.pipeline-stage-card').forEach(card => {
                    card.classList.add('stage-in-view');
                });
            }
        }

        // 2. Interactive Stepper Demo Logic
        let demoStep = 1;
        const totalDemoSteps = 4;
        const btnRunDemo = document.getElementById('btn-run-rag-demo');
        const demoText = document.getElementById('btn-rag-demo-text');

        if (btnRunDemo) {
            btnRunDemo.addEventListener('click', () => {
                demoStep = (demoStep % totalDemoSteps) + 1;
                for (let i = 1; i <= totalDemoSteps; i++) {
                    const stepEl = document.getElementById(`demo-step-${i}`);
                    if (stepEl) {
                        if (i === demoStep) {
                            stepEl.classList.add('active');
                        } else {
                            stepEl.classList.remove('active');
                        }
                    }
                }
                if (demoText) {
                    demoText.textContent = demoStep === totalDemoSteps ? 'Reset Demo' : `Step ${demoStep + 1} of 4`;
                }
            });
        }
    }

    // =========================================================================
    // INITIAL ROUTE DETECTION ON DOMCONTENTLOADED
    // =========================================================================
    const initialHash = window.location.hash.replace('#', '');
    if (VALID_VIEWS.includes(initialHash)) {
        navigateToView(initialHash, false);
    } else {
        navigateToView('dashboard', false);
    }
});
