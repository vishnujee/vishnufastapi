function toggleDeleteMethod() {
    const methodSelect = document.getElementById('deletePages-method');
    const visualInterface = document.getElementById('visual-method-interface');
    const textInterface = document.getElementById('text-method-interface');
    const fileInput = document.getElementById('deletePages-file');

    if (methodSelect.value === 'visual') {
        visualInterface.classList.remove('hidden');
        textInterface.classList.add('hidden');

        // Auto-load previews if file is already selected
        if (fileInput.files[0]) {
            console.log('Auto-loading previews for visual method...');
            // Ensure preview container exists
            ensurePreviewContainerExists();
            deletePDFPagesClientSide();
        }
    } else {
        visualInterface.classList.add('hidden');
        textInterface.classList.remove('hidden');
    }
}

// Helper function to ensure preview container exists
function ensurePreviewContainerExists() {
    let previewContainer = document.getElementById('delete-pages-preview-container');
    if (!previewContainer) {
        const visualInterface = document.getElementById('visual-method-interface');
        previewContainer = document.createElement('div');
        previewContainer.id = 'delete-pages-preview-container';
        visualInterface.appendChild(previewContainer);
    }
    return previewContainer;
}

// This function - it's referenced in HTML
function toggleDeleteInputs() {
    const deleteType = document.getElementById('deletePages-type');
    const specificInput = document.getElementById('specific-pages-input');
    const rangeInput = document.getElementById('range-pages-input');

    // Always show specific input (now supports both), hide range input
    specificInput.classList.remove('hidden');
    rangeInput.classList.add('hidden');

    // Update placeholder based on selection (optional)
    const inputField = document.getElementById('deletePages-specific');
    if (deleteType.value === 'specific') {
        inputField.placeholder = "Enter pages or ranges (e.g., 1,3,5-10,17)";
    } else {
        inputField.placeholder = "Enter page range (e.g., 5-10)";
    }
}

// Add this function - it's referenced in your HTML
function displayTotalPages(fileInputId, totalPagesId) {
    const fileInput = document.getElementById(fileInputId);
    const totalPagesElement = document.getElementById(totalPagesId);
    const fileNameElement = document.getElementById(`${fileInputId}-name`);

    if (fileInput.files[0]) {
        const file = fileInput.files[0];
        fileNameElement.textContent = file.name;

        // Load PDF to get total pages
        const fileReader = new FileReader();
        fileReader.onload = function () {
            const arrayBuffer = this.result;
            pdfjsLib.getDocument({ data: arrayBuffer }).promise.then(function (pdf) {
                totalPagesElement.textContent = `Total Pages: ${pdf.numPages}`;

                // Auto-load previews if visual method is selected AND file is valid
                const methodSelect = document.getElementById('deletePages-method');
                if (methodSelect && methodSelect.value === 'visual' && file.type === 'application/pdf') {
                    console.log('Auto-loading previews from displayTotalPages...');
                    ensurePreviewContainerExists();
                    deletePDFPagesClientSide();
                }
            }).catch(function (error) {
                console.error('Error loading PDF:', error);
                totalPagesElement.textContent = 'Total Pages: Error loading';
            });
        };
        fileReader.onerror = function () {
            console.error('Error reading file:', fileReader.error);
            totalPagesElement.textContent = 'Total Pages: Error reading file';
        };
        fileReader.readAsArrayBuffer(file);
    } else {
        fileNameElement.textContent = 'No file selected';
        totalPagesElement.textContent = 'Total Pages: Not loaded';
    }
}

// Add FileSaver.js functionality if not available
if (typeof saveAs === 'undefined') {
    function saveAs(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

async function deletePDFPagesClientSide() {
    console.log('Starting client-side PDF page deletion...');

    const fileInput = document.getElementById('deletePages-file');
    const resultDiv = document.getElementById('result-deletePagesForm');
    const progressDiv = document.getElementById('progress-deletePagesForm');
    const progressText = document.getElementById('progress-text-deletePagesForm');

    // Validation
    if (!fileInput || !fileInput.files[0]) {
        alert('Please select a PDF file.');
        return;
    }

    const file = fileInput.files[0];

    // File size validation
    if (file.size > 150 * 1024 * 1024) {
        alert("Use PDF less than 150 mb");
        return;
    }

    // Validate file type
    if (file.type !== 'application/pdf') {
        alert('Please select a PDF file.');
        return;
    }

    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Loading PDF previews...';

    try {
        // Load and display page previews
        await loadDeletePagePreviews(file);

    } catch (error) {
        console.error('Page preview loading failed:', error);
        resultDiv.innerHTML = `
            <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                <div class="flex items-center mb-2">
                    <i class="fas fa-times-circle text-red-500 text-xl mr-2"></i>
                    <span class="text-red-800 font-semibold text-lg">Preview Loading Failed</span>
                </div>
                <div class="text-red-700 text-sm">
                    Failed to load page previews. Please try the text input method instead.
                </div>
            </div>
        `;
        progressDiv.style.display = 'none';
    }
}

function handleFileSelectForDeletePages() {
    const fileInputId = 'deletePages-file';
    const totalPagesId = 'deletePages-total';

    // First update the file name and total pages
    displayTotalPages(fileInputId, totalPagesId);

    // Then check if we should auto-load previews
    const fileInput = document.getElementById(fileInputId);
    const methodSelect = document.getElementById('deletePages-method');

    if (fileInput.files[0] && methodSelect.value === 'visual') {
        console.log('File selected with visual method - loading previews...');
        // Ensure preview container exists
        ensurePreviewContainerExists();
        // Small delay to ensure displayTotalPages completes
        setTimeout(() => {
            deletePDFPagesClientSide();
        }, 100);
    }
}

// Updated preview loading function
async function loadDeletePagePreviews(file) {
    // Ensure preview container exists
    const previewContainer = ensurePreviewContainerExists();

    // Clear existing content but keep the container
    //  you can below button in below html
//     <button type="button" onclick="deletePDFPagesClientSide()" class="px-4 py-2 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 transition-colors">
//     <i class="fas fa-sync-alt mr-2"></i>Reload Previews
// </button>
    previewContainer.innerHTML = `
        <div class="mb-4">
            <h3 class="text-lg font-semibold text-gray-800 mb-3">Select Pages to Delete</h3>
            <p class="text-sm text-gray-600 mb-4">Click on pages to select/deselect them for deletion. Selected pages will be highlighted in red.</p>
            <div class="flex flex-wrap gap-2 mb-4">
                <button type="button" id="select-all-pages" class="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors">
                    <i class="fas fa-check-square mr-2"></i>Select All
                </button>
                <button type="button" id="deselect-all-pages" class="px-4 py-2 bg-gray-600 text-white text-sm rounded-lg hover:bg-gray-700 transition-colors">
                    <i class="fas fa-times-circle mr-2"></i>Deselect All
                </button>
              
                <button type="button" id="confirm-deletion" class="px-4 py-2 bg-red-600 text-white text-sm rounded-lg hover:bg-red-700 transition-colors ml-auto font-semibold">
                    <i class="fas fa-trash-alt mr-2"></i>Delete Selected Pages
                </button>
            </div>
            <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-4">
                <div class="flex items-center">
                    <i class="fas fa-exclamation-triangle text-yellow-500 mr-2"></i>
                    <span class="text-yellow-800 text-sm font-medium">Selected pages will be permanently deleted from the PDF</span>
                </div>
            </div>
            <div id="page-previews-grid" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4 max-h-96 overflow-y-auto p-4 border border-gray-200 rounded-lg bg-gray-50"></div>
        </div>
    `;

    const grid = document.getElementById('page-previews-grid');
    const progressText = document.getElementById('progress-text-deletePagesForm');

    try {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        const numPages = pdf.numPages;

        console.log(`Loading ${numPages} page previews for deletion selection...`);

        for (let i = 1; i <= numPages; i++) {
            const progress = Math.round((i / numPages) * 100);
            progressText.textContent = `Loading page ${i}/${numPages}... (${progress}%)`;

            const page = await pdf.getPage(i);
            const viewport = page.getViewport({ scale: 0.2 });

            const canvas = document.createElement('canvas');
            canvas.width = viewport.width;
            canvas.height = viewport.height;
            const context = canvas.getContext('2d');

            // White background
            context.fillStyle = 'white';
            context.fillRect(0, 0, canvas.width, canvas.height);

            // Render PDF page
            await page.render({
                canvasContext: context,
                viewport: viewport
            }).promise;

            const pageElement = document.createElement('div');
            pageElement.className = 'page-preview-item border-2 border-gray-300 rounded-lg p-3 bg-white cursor-pointer transition-all duration-200 hover:shadow-md';
            pageElement.dataset.pageNum = i;
            pageElement.title = `Click to select Page ${i} for deletion`;

            pageElement.innerHTML = `
                <div class="flex flex-col items-center">
                    <div class="canvas-container mb-2 border border-gray-200 bg-white overflow-hidden rounded shadow-sm">
                        <!-- Canvas will be appended here -->
                    </div>
                    <div class="flex items-center justify-between w-full mt-2">
                        <span class="text-gray-700 text-sm font-semibold">Page ${i}</span>
                        <div class="selection-indicator w-5 h-5 border-2 border-gray-400 rounded-full flex items-center justify-center bg-white">
                            <i class="fas fa-check text-white text-xs checkmark hidden"></i>
                        </div>
                    </div>
                </div>
            `;

            // Append canvas
            const canvasContainer = pageElement.querySelector('.canvas-container');
            canvasContainer.appendChild(canvas);

            // Add click handler
            pageElement.addEventListener('click', function () {
                const isSelected = this.classList.contains('border-red-500');

                if (isSelected) {
                    // Deselect
                    this.classList.remove('border-red-500', 'bg-red-50', 'shadow-lg');
                    this.classList.add('border-gray-300');
                    this.querySelector('.checkmark').classList.add('hidden');
                    this.querySelector('.selection-indicator').classList.remove('bg-red-500', 'border-red-500');
                } else {
                    // Select
                    this.classList.remove('border-gray-300');
                    this.classList.add('border-red-500', 'bg-red-50', 'shadow-lg');
                    this.querySelector('.checkmark').classList.remove('hidden');
                    this.querySelector('.selection-indicator').classList.add('bg-red-500', 'border-red-500');
                }

                updateSelectedPagesCount();
            });

            grid.appendChild(pageElement);
        }

        console.log(`Successfully loaded ${numPages} page previews`);
        progressText.textContent = 'Page previews loaded. Select pages to delete.';

        // Add event listeners for control buttons
        document.getElementById('select-all-pages').addEventListener('click', selectAllPages);
        document.getElementById('deselect-all-pages').addEventListener('click', deselectAllPages);
        // document.getElementById('confirm-deletion').addEventListener('click', () =>
        //     processSelectedPagesDeletion(file)
        // );
        const confirmButton = document.getElementById('confirm-deletion');
        // Remove any existing event listeners by cloning and replacing
        const newConfirmButton = confirmButton.cloneNode(true);
        confirmButton.parentNode.replaceChild(newConfirmButton, confirmButton);
        // Add fresh event listener
        newConfirmButton.addEventListener('click', () => processSelectedPagesDeletion(file));

        // Initialize selected pages count
        updateSelectedPagesCount();

    } catch (error) {
        console.error('Error loading page previews:', error);
        throw error;
    } finally {
        const progressDiv = document.getElementById('progress-deletePagesForm');
        progressDiv.style.display = 'none';
    }
}

// Text-based deletion processing
async function processTextBasedDeletion() {
    const fileInput = document.getElementById('deletePages-file');
    const deleteType = document.getElementById('deletePages-type');
    const specificInput = document.getElementById('deletePages-specific');
    const rangeInput = document.getElementById('deletePages-range');
    const resultDiv = document.getElementById('result-deletePagesForm');
    const progressDiv = document.getElementById('progress-deletePagesForm');
    const progressText = document.getElementById('progress-text-deletePagesForm');
    const submitButton = document.querySelector('#text-method-interface button');

    // Validation
    if (!fileInput || !fileInput.files[0]) {
        alert('Please select a PDF file.');
        return;
    }

    const file = fileInput.files[0];

    // File size validation
    if (file.size > 150 * 1024 * 1024) {
        alert("Use PDF less than 150 mb");
        return;
    }

    if (file.type !== 'application/pdf') {
        alert('Please select a PDF file.');
        return;
    }

    // Show progress early for PDF loading
    progressDiv.style.display = 'block';
    progressText.textContent = 'Loading PDF for validation...';
    submitButton.disabled = true;

    try {
        // Load PDF document FIRST to get total pages
        const pdfBytes = await file.arrayBuffer();
        const pdfDoc = await PDFLib.PDFDocument.load(pdfBytes);
        const totalPages = pdfDoc.getPageCount();

        let pagesToDelete = [];

        // Get input value (use either specific or range input based on which one is visible)
        // Always use the specific input since it now supports both formats
        let inputValue = specificInput.value;

        if (!inputValue) {
            alert('Please enter pages to delete (e.g., 1,3,5-10,17)');
            return;
        }

        // Parse combined input (supports both specific pages and ranges)
        const parts = inputValue.split(',');

        for (let part of parts) {
            part = part.trim();

            if (part.includes('-')) {
                // Handle range (e.g., "5-10")
                const rangeMatch = part.match(/^(\d+)-(\d+)$/);
                if (!rangeMatch) {
                    alert(`Invalid range format: "${part}". Use format like 5-10`);
                    return;
                }

                const start = parseInt(rangeMatch[1]);
                const end = parseInt(rangeMatch[2]);

                // Validate range
                if (start < 1 || end > totalPages || start > end) {
                    alert(`Invalid range ${start}-${end}. PDF has ${totalPages} pages.`);
                    return;
                }

                // Add all pages in range
                for (let i = start; i <= end; i++) {
                    pagesToDelete.push(i);
                }
            } else {
                // Handle single page (e.g., "1", "3", "17")
                const pageNum = parseInt(part);

                if (isNaN(pageNum) || pageNum < 1) {
                    alert(`Invalid page number: "${part}". Please enter valid numbers.`);
                    return;
                }

                if (pageNum > totalPages) {
                    alert(`Page ${pageNum} exceeds total pages (${totalPages}).`);
                    return;
                }

                pagesToDelete.push(pageNum);
            }
        }

        // Remove duplicates and sort
        pagesToDelete = [...new Set(pagesToDelete)].sort((a, b) => a - b);

        // Check if any pages to delete
        if (pagesToDelete.length === 0) {
            alert('No valid pages selected for deletion.');
            return;
        }

        // Convert to 0-based for PDF processing
        const pagesToDeleteZeroBased = pagesToDelete.map(page => page - 1);

        progressText.textContent = 'Processing deletion...';
        submitButton.innerHTML = '<i class="fas fa-trash-alt mr-2"></i> Deleting Pages...';

        console.log(`Deleting pages: ${pagesToDelete.join(', ')}`);

        // Create new PDF without deleted pages
        const newPdfDoc = await PDFLib.PDFDocument.create();
        const pageIndices = Array.from({ length: totalPages }, (_, i) => i)
            .filter(page => !pagesToDeleteZeroBased.includes(page));

        // Copy remaining pages
        if (pageIndices.length > 0) {
            const copiedPages = await newPdfDoc.copyPages(pdfDoc, pageIndices);
            copiedPages.forEach(page => newPdfDoc.addPage(page));
        } else {
            throw new Error('Cannot delete all pages from the PDF.');
        }

        progressText.textContent = 'Saving modified PDF...';

        // Save the modified PDF
        const modifiedPdfBytes = await newPdfDoc.save();
        const modifiedBlob = new Blob([modifiedPdfBytes], { type: 'application/pdf' });

        // Download the modified file
        const filename = `modified_${file.name.replace('.pdf', '')}_deleted_${pagesToDelete.length}_pages.pdf`;
        const url = URL.createObjectURL(modifiedBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        resultDiv.innerHTML = `
            <div class="bg-green-50 border border-green-200 rounded-lg p-4">
                <div class="flex items-center mb-2">
                    <i class="fas fa-check-circle text-green-500 text-xl mr-2"></i>
                    <span class="text-green-800 font-semibold text-lg">Pages Deleted Successfully!</span>
                </div>
                <div class="text-green-700 text-sm space-y-1">
                    <div><strong>File:</strong> ${filename}</div>
                    <div><strong>Pages deleted:</strong> ${pagesToDelete.length} (${pagesToDelete.join(', ')})</div>
                    <div><strong>Remaining pages:</strong> ${pageIndices.length}</div>
                </div>
            </div>
        `;

    } catch (error) {
        console.error('Page deletion failed:', error);
        resultDiv.innerHTML = `
            <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                <div class="flex items-center mb-2">
                    <i class="fas fa-times-circle text-red-500 text-xl mr-2"></i>
                    <span class="text-red-800 font-semibold text-lg">Page Deletion Failed</span>
                </div>
                <div class="text-red-700 text-sm">
                    ${error.message}
                </div>
            </div>
        `;
    } finally {
        progressDiv.style.display = 'none';
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-trash-alt mr-2"></i> Delete Pages (Text Method)';
    }
}

// Helper functions for page selection
function selectAllPages() {
    const pageElements = document.querySelectorAll('.page-preview-item');
    pageElements.forEach(element => {
        element.classList.remove('border-gray-300');
        element.classList.add('border-red-500', 'bg-red-50', 'shadow-lg');
        element.querySelector('.checkmark').classList.remove('hidden');
        element.querySelector('.selection-indicator').classList.add('bg-red-500', 'border-red-500');
    });
    updateSelectedPagesCount();
}

function deselectAllPages() {
    const pageElements = document.querySelectorAll('.page-preview-item');
    pageElements.forEach(element => {
        element.classList.remove('border-red-500', 'bg-red-50', 'shadow-lg');
        element.classList.add('border-gray-300');
        element.querySelector('.checkmark').classList.add('hidden');
        element.querySelector('.selection-indicator').classList.remove('bg-red-500', 'border-red-500');
    });
    updateSelectedPagesCount();
}

function updateSelectedPagesCount() {
    const confirmButton = document.getElementById('confirm-deletion');
    const selectedCount = document.querySelectorAll('.page-preview-item.border-red-500').length;
    const totalCount = document.querySelectorAll('.page-preview-item').length;

    if (selectedCount > 0) {
        confirmButton.innerHTML = `<i class="fas fa-trash-alt mr-2"></i>Delete ${selectedCount} of ${totalCount} Pages`;
        confirmButton.disabled = false;
        confirmButton.classList.remove('bg-gray-400', 'cursor-not-allowed');
        confirmButton.classList.add('bg-red-600', 'hover:bg-red-700', 'cursor-pointer');
    } else {
        confirmButton.innerHTML = `<i class="fas fa-trash-alt mr-2"></i>Delete Selected Pages`;
        confirmButton.disabled = true;
        confirmButton.classList.remove('bg-red-600', 'hover:bg-red-700', 'cursor-pointer');
        confirmButton.classList.add('bg-gray-400', 'cursor-not-allowed');
    }
}

function getSelectedPages() {
    const selectedElements = document.querySelectorAll('.page-preview-item.border-red-500');
    return Array.from(selectedElements).map(element =>
        parseInt(element.dataset.pageNum)
    ).sort((a, b) => a - b);
}

// Main processing function for selected pages
async function processSelectedPagesDeletion(file) {
    const progressDiv = document.getElementById('progress-deletePagesForm');
    const progressText = document.getElementById('progress-text-deletePagesForm');
    const resultDiv = document.getElementById('result-deletePagesForm');
    const submitButton = document.querySelector('#deletePagesForm button');

    const pagesToDelete = getSelectedPages();

    if (pagesToDelete.length === 0) {
        alert('Please select at least one page to delete.');
        return;
    }

    // Show confirmation dialog for large deletions
    if (pagesToDelete.length > 10) {
        const confirmDelete = confirm(`You are about to delete ${pagesToDelete.length} pages. This action cannot be undone. Continue?`);
        if (!confirmDelete) {
            return;
        }
    }

    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Processing deletion...';
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-trash-alt mr-2"></i> Deleting Pages...';

    try {
        // Load PDF document
        progressText.textContent = 'Loading PDF...';
        const pdfBytes = await file.arrayBuffer();
        const pdfDoc = await PDFLib.PDFDocument.load(pdfBytes);
        const totalPages = pdfDoc.getPageCount();

        // Validate page numbers
        if (pagesToDelete.some(page => page < 1 || page > totalPages)) {
            throw new Error(`Invalid page numbers. PDF has ${totalPages} pages.`);
        }

        console.log(`Deleting pages: ${pagesToDelete.join(', ')}`);

        // Create new PDF without deleted pages
        progressText.textContent = 'Creating new PDF...';
        const newPdfDoc = await PDFLib.PDFDocument.create();
        const pageIndices = Array.from({ length: totalPages }, (_, i) => i + 1)
            .filter(page => !pagesToDelete.includes(page))
            .map(page => page - 1); // Convert to 0-based

        // Copy remaining pages
        if (pageIndices.length > 0) {
            progressText.textContent = `Copying ${pageIndices.length} remaining pages...`;
            const copiedPages = await newPdfDoc.copyPages(pdfDoc, pageIndices);
            copiedPages.forEach(page => newPdfDoc.addPage(page));
        } else {
            throw new Error('Cannot delete all pages from the PDF.');
        }

        progressText.textContent = 'Saving modified PDF...';

        // Save the modified PDF
        const modifiedPdfBytes = await newPdfDoc.save();
        const modifiedBlob = new Blob([modifiedPdfBytes], { type: 'application/pdf' });

        // Download the modified file
        const filename = `modified_${file.name.replace('.pdf', '')}_deleted_${pagesToDelete.length}_pages.pdf`;
        const url = URL.createObjectURL(modifiedBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        // Show success message
        resultDiv.innerHTML = `
            <div class="bg-green-50 border border-green-200 rounded-lg p-4">
                <div class="flex items-center mb-2">
                    <i class="fas fa-check-circle text-green-500 text-xl mr-2"></i>
                    <span class="text-green-800 font-semibold text-lg">Pages Deleted Successfully!</span>
                </div>
                <div class="text-green-700 text-sm space-y-1">
                    <div><strong>File:</strong> ${filename}</div>
                    <div><strong>Pages deleted:</strong> ${pagesToDelete.length} (${pagesToDelete.join(', ')})</div>
                    <div><strong>Remaining pages:</strong> ${pageIndices.length}</div>
                    <div class="mt-2 text-green-600">
                        <i class="fas fa-info-circle mr-1"></i>
                        Pages ${pagesToDelete.join(', ')} were permanently removed from the PDF
                    </div>
                </div>
            </div>
        `;

        // Clear preview container but don't remove it
        // const previewContainer = document.getElementById('delete-pages-preview-container');
        // if (previewContainer) {
        //     previewContainer.innerHTML = '';
        // }

        const previewContainer = document.getElementById('delete-pages-preview-container');
if (previewContainer) {
    previewContainer.innerHTML = `
        <div class="text-center py-8">
            <div class="mb-4">
                <i class="fas fa-check-circle text-green-500 text-4xl"></i>
            </div>
            <h3 class="text-lg font-semibold text-gray-800 mb-2">Pages Deleted Successfully!</h3>
            <p class="text-gray-600 mb-4">If you want again Page preview , Click Reload Previews below.</p>
            <button type="button" onclick="deletePDFPagesClientSide()" 
                    class="px-4 py-2 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 transition-colors">
                <i class="fas fa-sync-alt mr-2"></i>Reload Previews
            </button>
        </div>
    `;
}

    } catch (error) {
        console.error('Page deletion failed:', error);
        resultDiv.innerHTML = `
            <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                <div class="flex items-center mb-2">
                    <i class="fas fa-times-circle text-red-500 text-xl mr-2"></i>
                    <span class="text-red-800 font-semibold text-lg">Page Deletion Failed</span>
                </div>
                <div class="text-red-700 text-sm">
                    ${error.message}<br>
                    <span class="text-red-600">
                        <i class="fas fa-sync-alt mr-1"></i>
                        Falling back to server processing...
                    </span>
                </div>
            </div>
        `;

        // Fallback to server processing
        setTimeout(() => {
            processPDF('delete_pdf_pages', 'deletePagesForm');
        }, 2000);

    } finally {
        progressDiv.style.display = 'none';
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-trash-alt mr-2"></i> Delete Pages';
    }
}

document.addEventListener('DOMContentLoaded', function () {
    const methodSelect = document.getElementById('deletePages-method');
    if (methodSelect) {
        methodSelect.addEventListener('change', toggleDeleteMethod);
        // Trigger initial setup
        toggleDeleteMethod();
    }

    // Check if there's already a file selected and visual method is chosen
    const fileInput = document.getElementById('deletePages-file');
    if (fileInput && fileInput.files[0] && methodSelect && methodSelect.value === 'visual') {
        console.log('Initial load with file and visual method - loading previews');
        ensurePreviewContainerExists();
        setTimeout(() => {
            deletePDFPagesClientSide();
        }, 500);
    }

    console.log('PDF tools initialized');
});