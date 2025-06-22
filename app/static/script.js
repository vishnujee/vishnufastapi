const BASE_URL = window.location.origin; // Adjust if backend API path differs

// Set PDF.js worker script
if (typeof pdfjsLib !== 'undefined') {
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
}

function updateFileSize() {
    const fileInput = document.getElementById('compress-file');
    const fileSizeDisplay = document.getElementById('original-file-size');
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        fileSizeDisplay.textContent = `Original File Size: ${sizeMB} MB`;
    } else {
        fileSizeDisplay.textContent = 'Original File Size: Not selected';
    }
}

async function computeAllCompressionSizes() {
    console.log('Computing all compression sizes');
    const form = document.getElementById('compressForm');
    const fileInput = form.querySelector('input[type="file"]');
    const resultDiv = document.getElementById('result-compressForm');
    const progressDiv = document.getElementById('progress-compressForm');
    const progressText = document.getElementById('progress-text-compressForm');
    const compressionResults = document.getElementById('compression-results');
    const compressionSizes = document.getElementById('compression-sizes');
    const computeButton = form.querySelector('button[onclick="computeAllCompressionSizes()"]');

    if (!fileInput.files.length) {
        resultDiv.textContent = 'Please select a PDF file.';
        resultDiv.classList.add('text-red-600');
        return;
    }

    const file = fileInput.files[0];
    const sizeMB = file.size / (1024 * 1024);
    if (sizeMB > 160) {
        resultDiv.textContent = `File ${file.name} exceeds 160MB limit.`;
        resultDiv.classList.add('text-red-600');
        return;
    }
    if (file.type !== 'application/pdf') {
        resultDiv.textContent = `File ${file.name} must be a PDF.`;
        resultDiv.classList.add('text-red-600');
        return;
    }

    const customDpi = form.querySelector('input[name="custom_dpi"]').value;
    const customQuality = form.querySelector('input[name="custom_quality"]').value;
    if (customDpi && customQuality) {
        const dpi = parseInt(customDpi);
        console.log('Custom DPI:', dpi);
        const quality = parseInt(customQuality);
        if (dpi < 50 || dpi > 250 || quality < 10 || quality > 100) {
            resultDiv.textContent = 'Invalid custom DPI (50-400) or quality (10-100).';
            resultDiv.classList.add('text-red-600');
            return;
        }
    }

    resultDiv.textContent = '';
    resultDiv.classList.remove('text-red-600', 'text-green-600');
    progressDiv.style.display = 'block';
    progressText.textContent = 'Estimating sizes...';
    computeButton.disabled = true;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('custom_dpi', customDpi || '180');
    formData.append('custom_quality', customQuality || '50');

    try {
        const response = await fetch(`${BASE_URL}/estimate_compression_sizes`, {
            method: 'POST',
            body: formData
        });

        progressDiv.style.display = 'none';
        computeButton.disabled = false;

        if (response.ok) {
            const sizes = await response.json();
            compressionSizes.innerHTML = `
                <li>High Compression (72 DPI, 20% Quality): ${sizes.high.toFixed(2)} MB</li>
                <li>Medium Compression (100 DPI, 30% Quality): ${sizes.medium.toFixed(2)} MB</li>
                <li>Low Compression (120 DPI, 40% Quality): ${sizes.low.toFixed(2)} MB</li>
                <li>Custom Compression (${customDpi || 180} DPI, ${customQuality || 50}% Quality): ${sizes.custom.toFixed(2)} MB</li>
            `;
            compressionResults.classList.remove('hidden');
            resultDiv.textContent = 'Estimated sizes calculated successfully!';
            resultDiv.classList.add('text-green-600');
        } else {
            const error = await response.json();
            console.error('Estimation error:', error);
            resultDiv.textContent = `Error: ${error.detail || 'Unknown error'}`;
            resultDiv.classList.add('text-red-600');
        }
    } catch (e) {
        console.error('Fetch error:', e);
        progressDiv.style.display = 'none';
        computeButton.disabled = false;
        resultDiv.textContent = `Error: ${e.message}. Please check the server logs.`;
        resultDiv.classList.add('text-red-600');
    }
}

// Add slider value display update
function initSliders() {
    const dpiSlider = document.getElementById('custom_dpi');
    const qualitySlider = document.getElementById('custom_quality');
    const dpiValue = document.getElementById('dpi-value');
    const qualityValue = document.getElementById('quality-value');

    if (dpiSlider && dpiValue) {
        dpiSlider.addEventListener('input', () => {
            dpiValue.textContent = dpiSlider.value;
        });
    }
    if (qualitySlider && qualityValue) {
        qualitySlider.addEventListener('input', () => {
            qualityValue.textContent = qualitySlider.value;
        });
    }
}

// Toggle between specific pages and range inputs
function toggleDeleteInputs() {
    const deleteType = document.getElementById('deletePages-type').value;
    document.getElementById('specific-pages-input').classList.toggle('hidden', deleteType !== 'specific');
    document.getElementById('range-pages-input').classList.toggle('hidden', deleteType !== 'range');
}

// Existing toggleCustomInputs
function toggleCustomInputs() {
    const preset = document.getElementById('compress-preset').value;
    const customOptions = document.getElementById('custom-compress-options');
    customOptions.classList.toggle('hidden', preset !== 'Custom');
}

// Helper function to get total pages
async function getTotalPages(file) {
    if (!file || typeof pdfjsLib === 'undefined') return 0;
    try {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        return pdf.numPages;
    } catch (err) {
        console.error('Error fetching total pages:', err);
        return 0;
    }
}

// document.getElementById('description-position').addEventListener('change', function() {
//     const customContainer = document.getElementById('custom-position-container');
//     customContainer.classList.toggle('hidden', this.value !== 'custom');
//   });


  async function processPDF(endpoint, formId) {
    console.log(`Processing PDF for endpoint: ${endpoint}, form: ${formId}`);
    const form = document.getElementById(formId);
    const resultDiv = document.getElementById(`result-${formId}`);
    const submitButton = form.querySelector('button');
    const progressDiv = document.getElementById(`progress-${formId}`);
    const progressText = document.getElementById(`progress-text-${formId}`);
    const spinnerStyle = document.createElement('style');

    if (!form) {
        console.error(`Form with ID ${formId} not found`);
        resultDiv.textContent = 'Form not found.';
        resultDiv.classList.add('text-red-600');
        return;
    }

    let formData = new FormData();

    if (endpoint === 'convert_image_to_pdf') {
        const fileInput = form.querySelector('#imageToPdf-file');
        const description = form.querySelector('#image-description')?.value || '';
        const position = form.querySelector('#description-position')?.value || 'bottom';
        const fontSize = parseInt(form.querySelector('#description-font-size')?.value || '12');
        const pageSize = form.querySelector('select[name="page_size"]')?.value || 'A4';
        const orientation = form.querySelector('select[name="orientation"]')?.value || 'Portrait';
        const customX = form.querySelector('#custom-x')?.value;
        const customY = form.querySelector('#custom-y')?.value;

        if (!fileInput || !fileInput.files[0]) {
            resultDiv.textContent = 'Please select an image file.';
            resultDiv.classList.add('text-red-600');
            return;
        }

        formData.append('file', fileInput.files[0]);
        formData.append('description', description);
        formData.append('description_position', position);
        formData.append('description_font_size', fontSize);
        formData.append('page_size', pageSize);
        formData.append('orientation', orientation);

        if (position === 'custom') {
            if (!customX || !customY || isNaN(customX) || isNaN(customY)) {
                resultDiv.textContent = 'Custom position requires valid X and Y coordinates.';
                resultDiv.classList.add('text-red-600');
                return;
            }
            formData.append('custom_x', parseFloat(customX));
            formData.append('custom_y', parseFloat(customY));
        }
    } else if (endpoint === 'compress_pdf') {
        const fileInput = form.querySelector('input[type="file"]');
        const preset = form.querySelector('select[name="preset"]').value;
        if (fileInput && fileInput.files[0]) {
            formData.append('file', fileInput.files[0]);
        }
        formData.append('preset', preset);
        if (preset === 'Custom') {
            const customDpi = form.querySelector('input[name="custom_dpi"]').value;
            const customQuality = form.querySelector('input[name="custom_quality"]').value;
            formData.append('custom_dpi', customDpi);
            formData.append('custom_quality', customQuality);
        }
    } else if (endpoint === 'delete_pdf_pages') {
        const deleteType = form.querySelector('select[name="delete_type"]').value;
        const fileInput = form.querySelector('input[type="file"]');
        if (!fileInput || !fileInput.files[0]) {
            resultDiv.textContent = 'Please select a PDF file.';
            resultDiv.classList.add('text-red-600');
            return;
        }
        const file = fileInput.files[0];
        let pages;
        if (deleteType === 'specific') {
            pages = form.querySelector('input[name="pages"]').value;
        } else {
            const range = form.querySelector('input[name="range"]').value;
            const totalPages = await getTotalPages(file);
            pages = expandPageRange(range, totalPages);
        }
        formData.append('file', file);
        formData.append('pages', pages);
    } else {
        const fileInput = form.querySelector('input[type="file"]');
        if (fileInput && fileInput.files[0]) {
            formData.append('file', fileInput.files[0]);
        }
        const conversionTypeInput = form.querySelector('input[name="conversionType"]:checked');
        if (conversionTypeInput) {
            formData.append('conversion_type', conversionTypeInput.value);
        }
    }

    if (!(await validateForm(form, endpoint, resultDiv))) {
        console.log('Validation failed');
        return;
    }

    console.log('Sending request to:', `${BASE_URL}/${endpoint}`);
    console.log('FormData contents:');
    for (const [key, value] of formData.entries()) {
        console.log(`${key}: ${value instanceof File ? value.name : value}`);
    }

    submitButton.disabled = true;
    progressDiv.style.display = 'block';
    progressText.textContent = 'Preparing files...';
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress = Math.min(progress + 10, 90);
        progressDiv.querySelector('progress').value = progress;
        progressText.textContent = `Uploading & Processing... ${progress}%`;
    }, 200);

    try {
        const response = await fetch(`${BASE_URL}/${endpoint}`, {
            method: 'POST',
            body: formData
        });
        clearInterval(progressInterval);
        progressDiv.querySelector('progress').value = 100;

        spinnerStyle.textContent = `
            .spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(235, 13, 13, 0.99);
                border-radius: 50%;
                border-top-color: rgba(37, 230, 20, 0.99);
                animation: spin 1s ease-in-out infinite;
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(spinnerStyle);
        progressText.innerHTML = 'Processing completed. Wait for download... <i class="fas fa-spinner fa-spin" style="font-size: 1rem;color: red;"></i>';

        if (response.ok) {
            const blob = await response.blob();
            const contentDisposition = response.headers.get('Content-Disposition');
            let filename = endpoint === 'compress_pdf' ? 'compressed.pdf' : 'output.pdf';
            if (contentDisposition) {
                const match = contentDisposition.match(/filename="(.+)"|filename=([^;]+)/i);
                if (match) filename = match[1] || match[2];
            }
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            window.URL.revokeObjectURL(url);
            resultDiv.textContent = endpoint === 'compress_pdf' ? 'PDF compressed successfully!' : 'Processing completed successfully!';
            resultDiv.classList.remove('text-red-600');
            resultDiv.classList.add('text-green-600');
        } else {
            const error = await response.json();
            let errorMessage = 'Unknown error';
            if (Array.isArray(error.detail)) {
                errorMessage = error.detail.map(err => `${err.loc.join('.')}: ${err.msg}`).join('; ');
            } else if (error.detail) {
                errorMessage = error.detail;
            }
            console.error('Backend error:', error);
            resultDiv.textContent = `Error: ${errorMessage}`;
            resultDiv.classList.remove('text-green-600');
            resultDiv.classList.add('text-red-600');
        }
    } catch (e) {
        clearInterval(progressInterval);
        console.error('Fetch error:', e, 'Endpoint:', endpoint);
        resultDiv.textContent = `Error: ${e.message}. Please check the server logs.`;
        resultDiv.classList.remove('text-green-600');
        resultDiv.classList.add('text-red-600');
    } finally {
        submitButton.disabled = false;
        setTimeout(() => {
            progressDiv.style.display = 'none';
            progressText.textContent = '';
        }, 2000);
    }
}

function updateFileOrder(files) {
    const fileOrder = Array.from(files).map(file => file.dataset.fileIndex);
    document.getElementById('merge-file-order').value = fileOrder.join(',');
}

function updateFileButtonStates(files) {
    files.forEach((file, index) => {
        const upButton = file.querySelector('.move-up');
        const downButton = file.querySelector('.move-down');
        upButton.disabled = index === 0;
        downButton.disabled = index === files.length - 1;
    });
}

// Modified validateForm
async function validateForm(form, endpoint, resultDiv) {
    const filesInput = form.querySelector('input[type="file"]');
    if (!filesInput) {
        resultDiv.textContent = 'File input not found in the form.';
        resultDiv.classList.add('text-red-600');
        console.error(`No input[type="file"] found in form ${form.id}`);
        return false;
    }
    const files = filesInput.files;
    const password = form.querySelector('input[type="password"]');
    const pages = form.querySelector('input[name="pages"]');
    const pageOrderInput = form.querySelector('input[name="page_order"]');
    const fileOrderInput = form.querySelector('input[name="file_order"]');

    if (!files.length) {
        resultDiv.textContent = 'Please select a file.';
        resultDiv.classList.add('text-red-600');
        return false;
    }

    if (endpoint === 'convert_image_to_pdf') {
        const file = files[0];
        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > 50) {
            resultDiv.textContent = `File ${file.name} exceeds 50MB limit.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (!['image/png', 'image/jpeg'].includes(file.type)) {
            resultDiv.textContent = `File ${file.name} must be PNG or JPEG.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const position = form.querySelector('#description-position')?.value;
        if (!position) {
            resultDiv.textContent = 'Description position field is missing or invalid.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const validPositions = ['top', 'bottom', 'top-left', 'top-center', 'top-right', 
                               'bottom-left', 'bottom-center', 'bottom-right', 'custom'];
        if (!validPositions.includes(position)) {
            resultDiv.textContent = `Invalid description position: ${position}.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (position === 'custom') {
            const customXInput = form.querySelector('#custom-x');
            const customYInput = form.querySelector('#custom-y');
            if (!customXInput || !customYInput) {
                resultDiv.textContent = 'Custom X and Y coordinate inputs are missing.';
                resultDiv.classList.add('text-red-600');
                return false;
            }
            const customX = customXInput.value;
            const customY = customYInput.value;
            if (!customX || !customY || isNaN(customX) || isNaN(customY)) {
                resultDiv.textContent = 'Custom position requires valid X and Y coordinates.';
                resultDiv.classList.add('text-red-600');
                return false;
            }
            const x = parseFloat(customX);
            const y = parseFloat(customY);
            if (x < 0 || y < 0) {
                resultDiv.textContent = 'X and Y coordinates must be non-negative.';
                resultDiv.classList.add('text-red-600');
                return false;
            }
        }
        const fontSize = form.querySelector('#description-font-size')?.value;
        if (!fontSize || isNaN(fontSize) || fontSize < 8 || fontSize > 72) {
            resultDiv.textContent = 'Font size must be a number between 8 and 72.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const pageSize = form.querySelector('select[name="page_size"]')?.value || 'A4';
        if (!['A4', 'Letter'].includes(pageSize)) {
            resultDiv.textContent = `Invalid page size: ${pageSize}. Choose A4 or Letter.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const orientation = form.querySelector('select[name="orientation"]')?.value || 'Portrait';
        if (!['Portrait', 'Landscape'].includes(orientation)) {
            resultDiv.textContent = `Invalid orientation: ${orientation}. Choose Portrait or Landscape.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
    }
    else if (endpoint === 'add_signature') {
        const pdfFile = files[0];
        const signatureFile = form.querySelector('input[name="signature_file"]').files[0];
        const selectedPages = form.querySelector('input[name="specific_pages"]').value;
        const size = form.querySelector('select[name="size"]').value;
        const position = form.querySelector('select[name="position"]').value;
        const alignment = form.querySelector('select[name="alignment"]').value;

        // Validate PDF file
        const pdfSizeMB = pdfFile.size / (1024 * 1024);
        if (pdfSizeMB > 50) {
            resultDiv.textContent = `PDF file ${pdfFile.name} exceeds 50MB limit.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (pdfFile.type !== 'application/pdf') {
            resultDiv.textContent = `File ${pdfFile.name} must be a PDF.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }

        // Validate signature file
        const sigSizeMB = signatureFile.size / (1024 * 1024);
        if (sigSizeMB > 10) {
            resultDiv.textContent = `Signature file ${signatureFile.name} exceeds 10MB limit.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (!['image/png', 'image/jpeg', 'image/jpg'].includes(signatureFile.type)) {
            resultDiv.textContent = `Signature file ${signatureFile.name} must be PNG or JPEG.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }

        // Validate selected pages
        if (!selectedPages) {
            resultDiv.textContent = 'Please select at least one page to sign.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (!/^[1-9]\d*(,[1-9]\d*)*$/.test(selectedPages)) {
            resultDiv.textContent = 'Invalid page selection format. Pages must be comma-separated positive integers.';
            resultDiv.classList.add('text-red-600');
            return false;
        }

        // Validate size, position, alignment
        if (!['small', 'medium', 'large'].includes(size)) {
            resultDiv.textContent = 'Invalid size. Choose small, medium, or large.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (!['top', 'center', 'bottom'].includes(position)) {
            resultDiv.textContent = 'Invalid position. Choose top, center, or bottom.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (!['left', 'center', 'right'].includes(alignment)) {
            resultDiv.textContent = 'Invalid alignment. Choose left, center, or right.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
    } else if (endpoint === 'merge_pdf') {
        if (files.length < 2) {
            resultDiv.textContent = 'Please select at least 2 PDF files.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        let totalSize = 0;
        for (let file of files) {
            const sizeMB = file.size / (1024 * 1024);
            totalSize += sizeMB;
            if (sizeMB > 30) {
                resultDiv.textContent = `File ${file.name} exceeds 30MB limit.`;
                resultDiv.classList.add('text-red-600');
                return false;
            }
        }
        const method = form.querySelector('select[name="method"]').value;
        const maxFiles = method === 'PyPDF2' ? 51 : 30;
        const maxSizeMB = method === 'PyPDF2' ? 90 : 50;
        if (files.length > maxFiles) {
            resultDiv.textContent = `Too many files. Maximum is ${maxFiles} for ${method}.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (totalSize > maxSizeMB) {
            resultDiv.textContent = `Total file size (${totalSize.toFixed(1)}MB) exceeds ${maxSizeMB}MB limit.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        // Validate file order
        if (!fileOrderInput || !fileOrderInput.value) {
            resultDiv.textContent = 'Please load and order files.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        // Validate file order if provided
        if (fileOrderInput && fileOrderInput.value) {
            const fileOrder = fileOrderInput.value.split(',').map(i => parseInt(i.trim()));
            if (fileOrder.length !== files.length || !fileOrder.every(i => i >= 0 && i < files.length)) {
                resultDiv.textContent = 'Invalid file order. Ensure all files are included in the order.';
                resultDiv.classList.add('text-red-600');
                return false;
            }
        }
    } else if (endpoint === 'add_page_numbers') {
        const file = files[0];
        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > 50) {
            resultDiv.textContent = `File ${file.name} exceeds 50MB limit.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (file.type !== 'application/pdf') {
            resultDiv.textContent = `File ${file.name} must be a PDF.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const position = form.querySelector('select[name="position"]').value;
        const alignment = form.querySelector('select[name="alignment"]').value;
        const format = form.querySelector('select[name="format"]').value;
        if (!['top', 'bottom'].includes(position)) {
            resultDiv.textContent = 'Invalid position. Choose top or bottom.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (!['left', 'center', 'right'].includes(alignment)) {
            resultDiv.textContent = 'Invalid alignment. Choose left, center, or right.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (!['page_x', 'x'].includes(format)) {
            resultDiv.textContent = 'Invalid format. Choose Page X or X.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
    } else if (endpoint === 'reorder_pages') {
        const file = files[0];
        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > 50) {
            resultDiv.textContent = `File ${file.name} exceeds 50MB limit.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (!pageOrderInput || !pageOrderInput.value) {
            resultDiv.textContent = 'Please load and reorder pages.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        // Validate page order format
        const pageOrder = pageOrderInput.value.split(',').map(p => p.trim());
        if (!pageOrder.every(p => /^[1-9]\d*$/.test(p))) {
            resultDiv.textContent = 'Invalid page order format. Use comma-separated positive integers.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
    } else if (endpoint === 'compress_pdf') {
        const file = files[0];
        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > 160) {
            resultDiv.textContent = `File ${file.name} exceeds 160MB limit.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (file.type !== 'application/pdf') {
            resultDiv.textContent = `File ${file.name} must be a PDF.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const preset = form.querySelector('select[name="preset"]').value;
        if (!['High', 'Medium', 'Low', 'Custom'].includes(preset)) {
            resultDiv.textContent = 'Invalid preset. Choose High, Medium, Low, or Custom.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (preset === 'Custom') {
            const customDpi = form.querySelector('input[name="custom_dpi"]').value;
            const customQuality = form.querySelector('input[name="custom_quality"]').value;
            if (!customDpi || !customQuality) {
                resultDiv.textContent = 'Custom preset requires DPI and quality values.';
                resultDiv.classList.add('text-red-600');
                return false;
            }
            const dpi = parseInt(customDpi);
            const quality = parseInt(customQuality);
            if (dpi < 50 || dpi > 400 || quality < 10 || quality > 100) {
                resultDiv.textContent = 'Invalid custom DPI (50-400) or quality (10-100).';
                resultDiv.classList.add('text-red-600');
                return false;
            }
        }
    } else if (endpoint === 'delete_pdf_pages') {
        const deleteTypeElement = form.querySelector('select[name="delete_type"]');
        if (!deleteTypeElement) {
            resultDiv.textContent = 'Delete type selector not found.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const deleteType = deleteTypeElement.value;
        let pagesInput = deleteType === 'specific' ? form.querySelector('input[name="pages"]').value : form.querySelector('input[name="range"]').value;
        console.log('deleteType:', deleteType, 'pagesInput:', pagesInput);
        let totalPages = 0;
        const file = form.querySelector('input[type="file"]').files[0];
        if (!file) {
            resultDiv.textContent = 'Please select a PDF file.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > 50) {
            resultDiv.textContent = `File ${file.name} exceeds 50MB limit.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (file.type !== 'application/pdf') {
            resultDiv.textContent = `File ${file.name} must be a PDF.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (typeof pdfjsLib === 'undefined') {
            resultDiv.textContent = 'PDF processing library not loaded. Please try again later.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        try {
            const arrayBuffer = await file.arrayBuffer();
            const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
            totalPages = pdf.numPages;
        } catch (err) {
            console.error('PDF page count error:', err);
            resultDiv.textContent = 'Error loading PDF file. Please ensure it’s a valid PDF.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (!pagesInput) {
            resultDiv.textContent = 'Please enter pages to delete.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (deleteType !== 'specific' && deleteType !== 'range') {
            resultDiv.textContent = 'Invalid delete type selected.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (deleteType === 'specific') {
            const pageNumbers = [...new Set(pagesInput.split(',').map(p => p.trim()).filter(p => p))];
            if (!pageNumbers.length || !pageNumbers.every(p => /^[1-9]\d*$/.test(p) && parseInt(p) <= totalPages)) {
                resultDiv.textContent = `Invalid page numbers. Use comma-separated integers (e.g., 1,12,32) within 1-${totalPages}.`;
                resultDiv.classList.add('text-red-600');
                return false;
            }
            form.querySelector('input[name="pages"]').value = pageNumbers.join(',');
        } else {
            const expandedPages = expandPageRange(pagesInput, totalPages);
            if (!expandedPages) {
                resultDiv.textContent = `Invalid range. Use format like 1-5 within 1-${totalPages}.`;
                resultDiv.classList.add('text-red-600');
                return false;
            }
            form.querySelector('input[name="pages"]').value = expandedPages;
        }
    } else {
        const maxSizeMB = endpoint === 'compress_pdf' ? 55 : endpoint === 'split_pdf' ? 100 : 50;
        const file = files[0];
        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > maxSizeMB) {
            resultDiv.textContent = `File ${file.name} exceeds ${maxSizeMB}MB limit.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
    }

    if ((endpoint === 'encrypt_pdf' || endpoint === 'remove_pdf_password') && (!password || !password.value)) {
        resultDiv.textContent = 'Please enter a password.';
        resultDiv.classList.add('text-red-600');
        return false;
    }

    return true;
}

// Display total pages in PDF
async function displayTotalPages(fileInputId, totalPagesId) {
    const fileInput = document.getElementById(fileInputId);
    const totalPagesDiv = document.getElementById(totalPagesId);
    const resultDiv = document.getElementById('result-deletePagesForm');
    if (!fileInput.files[0]) {
        totalPagesDiv.textContent = 'Total Pages: Not loaded';
        return;
    }
    try {
        const file = fileInput.files[0];
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        totalPagesDiv.textContent = `Total Pages: ${pdf.numPages}`;
        resultDiv.textContent = '';
    } catch (err) {
        console.error('Error counting pages:', err);
        totalPagesDiv.textContent = 'Total Pages: Error loading PDF';
        resultDiv.textContent = 'Invalid PDF file.';
        resultDiv.classList.add('text-red-600');
    }
}

// Expand page range (e.g., "1-5" to "1,2,3,4,5")
function expandPageRange(range, totalPages) {
    if (!range || typeof range !== 'string') {
        console.error('Invalid range input:', range);
        return null;
    }
    const trimmedRange = range.trim();
    const match = trimmedRange.match(/^(\d+)-(\d+)$/);
    if (!match) {
        console.error('Range format invalid:', trimmedRange);
        return null;
    }
    const start = parseInt(match[1]);
    const end = parseInt(match[2]);
    if (isNaN(start) || isNaN(end) || start < 1 || end < start || end > totalPages) {
        console.error(`Invalid range values: start=${start}, end=${end}, totalPages=${totalPages}`);
        return null;
    }
    return Array.from({ length: end - start + 1 }, (_, i) => start + i).join(',');
}

function formatResponse(text) {
    if (!text) return '';

    // Convert markdown tables to HTML tables
    text = text.replace(/(\|[^\n]+\|\r?\n\|[-: |]+\|\r?\n)((?:\|[^\n]+\|\r?\n?)+)/g, function (match, header, body) {
        let html = '<div class="overflow-x-auto"><table class="w-full border-collapse my-3">';

        // Process header
        const headers = header.split('|').slice(1, -1).map(h => h.trim());
        html += '<thead><tr class="bg-gray-100">';
        headers.forEach(h => {
            html += `<th class="p-2 border text-left">${h}</th>`;
        });
        html += '</tr></thead><tbody>';

        // Process body
        const rows = body.trim().split('\n');
        rows.forEach(row => {
            const cells = row.split('|').slice(1, -1).map(c => c.trim());
            html += '<tr>';
            cells.forEach(cell => {
                html += `<td class="p-2 border">${formatMarkdownInline(cell)}</td>`;
            });
            html += '</tr>';
        });

        html += '</tbody></table></div>';
        return html;
    });

    // Convert markdown lists
    text = text.replace(/^([*-]|\d+\.)\s+(.+)$/gm, function (match, bullet, content) {
        return `<li class="ml-5">${formatMarkdownInline(content)}</li>`;
    });

    // Wrap consecutive list items in ul/ol
    text = text.replace(/(<li>.*<\/li>)+/g, function (match) {
        const listType = match.includes('<li class="ml-5">') ? 'ul' : 'ol';
        return `<${listType} class="list-disc pl-5 my-2">${match}</${listType}>`;
    });

    // Convert markdown bold/italic
    text = formatMarkdownInline(text);

    // Convert line breaks to <br> for non-table, non-list content
    text = text.replace(/\n/g, function (match, offset, fullText) {
        if (!fullText.substring(offset).match(/^\n*(<table|<ul|<ol)/) &&
            !fullText.substring(0, offset).match(/(<\/table|<\/ul|<\/ol>)\n*$/)) {
            return '<br>';
        }
        return match;
    });

    // Convert markdown links
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-blue-600 hover:underline" target="_blank">$1</a>');

    return text;
}

function formatMarkdownInline(text) {
    if (!text) return '';

    // Bold
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italic
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Inline code
    text = text.replace(/`(.*?)`/g, '<code class="bg-gray-100 px-1 rounded">$1</code>');

    return text;
}

async function typewriterEffect(element, text, speed = 20) {
    // First format the response to handle markdown
    const formattedText = formatResponse(text);

    // Create a temporary element to parse the HTML
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = formattedText;

    // Process each node for typewriter effect
    element.innerHTML = '';
    await typewriterProcessNodes(element, tempDiv.childNodes, speed);
}

async function typewriterProcessNodes(parent, nodes, speed) {
    for (const node of nodes) {
        if (node.nodeType === Node.TEXT_NODE) {
            await typewriterAddText(parent, node.textContent, speed);
        } else if (node.nodeType === Node.ELEMENT_NODE) {
            // For tables, add them immediately
            if (node.tagName === 'TABLE') {
                const clone = node.cloneNode(true);
                parent.appendChild(clone);
            } else {
                const newElement = document.createElement(node.tagName);

                // Copy all attributes
                for (const attr of node.attributes) {
                    newElement.setAttribute(attr.name, attr.value);
                }

                // Copy classes
                if (node.className) {
                    newElement.className = node.className;
                }

                parent.appendChild(newElement);

                // Process child nodes recursively
                await typewriterProcessNodes(newElement, node.childNodes, speed);
            }
        }
    }
}

async function typewriterAddText(element, text, speed) {
    for (let i = 0; i < text.length; i++) {
        // Add the next character
        element.textContent += text[i];

        // Add blinking cursor (except for the last character)
        if (i < text.length - 1) {
            const cursor = document.createElement('span');
            cursor.className = 'blinking-cursor';
            cursor.textContent = '|';
            element.appendChild(cursor);

            // Wait for a bit
            await new Promise(resolve => setTimeout(resolve, speed));

            // Remove the cursor
            element.removeChild(cursor);
        }
    }
}

async function sendChat() {
    const chatInput = document.getElementById('chatInput').value.trim();
    const chatOutput = document.getElementById('chatOutput');
    const progressDiv = document.getElementById('progress-chat');
    const progressText = document.getElementById('progress-text-chat');

    if (!chatInput) {
        chatOutput.innerHTML = '<p class="text-red-600">Please enter a query.</p>';
        return;
    }

    progressDiv.style.display = 'block';
    progressText.textContent = 'Processing query...';

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: `query=${encodeURIComponent(chatInput)}&typewriter=true`
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Chat error');
        }

        const data = await response.json();

        // Create new message element
        const messageDiv = document.createElement('div');
        messageDiv.className = 'mb-4';

        // Add user query
        const userQuery = document.createElement('p');
        userQuery.className = 'font-semibold text-blue-600';
        userQuery.textContent = `You: ${chatInput}`;
        messageDiv.appendChild(userQuery);

        // Add AI response container
        const aiResponse = document.createElement('div');
        aiResponse.className = 'ai-response bg-gray-50 p-3 rounded mt-1';
        messageDiv.appendChild(aiResponse);

        // Insert at the top of chat output
        chatOutput.insertBefore(messageDiv, chatOutput.firstChild);

        // Clear input
        document.getElementById('chatInput').value = '';

        // Apply typewriter effect
        await typewriterEffect(aiResponse, data.answer);
    } catch (error) {
        console.error("Chat error:", error);
        const errorDiv = document.createElement('div');
        errorDiv.className = 'text-red-600 mb-4';
        errorDiv.innerHTML = `
            <p>Error: ${error.message}</p>
            <p class="text-sm text-gray-500">Please try again or refresh the page.</p>
        `;
        chatOutput.insertBefore(errorDiv, chatOutput.firstChild);
    } finally {
        progressDiv.style.display = 'none';
        progressText.textContent = '';
    }
}

function showTool(toolId) {
    localStorage.setItem('lastTool', toolId);
    // Hide all tool sections
    document.querySelectorAll('.tool-section').forEach(section => {
        section.style.display = 'none';
    });

    // Show selected tool
    document.getElementById(toolId).style.display = 'block';

    // Revert all nav-link styles
    document.querySelectorAll('.nav-link, .dropdown-content a, #mobile-submenu a').forEach(link => {
        link.classList.remove('text-green-600', 'font-bold');
    });

    // Highlight the clicked item
    if (event.currentTarget) {
        event.currentTarget.classList.add('text-green-600', 'font-bold');
    }

    // On mobile, hide the menus
    if (window.innerWidth <= 768) {
        const mobileMenu = document.getElementById('mobile-menu');
        const mobileSubmenu = document.getElementById('mobile-submenu');
        const menuButton = document.getElementById('mobile-menu-button');

        if (mobileSubmenu && !mobileSubmenu.classList.contains('hidden')) {
            mobileSubmenu.classList.add('hidden');
        }

        if (mobileMenu && !mobileMenu.classList.contains('hidden')) {
            mobileMenu.classList.add('hidden');
        }

        if (menuButton) {
            menuButton.querySelector('i').classList.remove('fa-times');
            menuButton.querySelector('i').classList.add('fa-bars');
        }
    }

    // Scroll to section
    document.getElementById(toolId).scrollIntoView({ behavior: 'smooth' });
}

async function processImage(endpoint, formId) {
    const form = document.getElementById(formId);
    const resultDiv = document.getElementById(`result-${formId}`);
    const progress = document.getElementById(`progress-${formId}`);
    const progressText = document.getElementById(`progress-text-${formId}`);
    const fileInput = form.querySelector('input[type="file"]');

    if (!fileInput.files[0]) {
        resultDiv.textContent = 'Please select an image file.';
        resultDiv.classList.add('text-red-600');
        return;
    }

    const formData = new FormData(form);
    progress.style.display = 'block';
    progressText.textContent = 'Processing...';
    form.querySelector('button').disabled = true;

    try {
        const response = await fetch(`/${endpoint}`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'processed_image.png';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        resultDiv.textContent = 'Background removed successfully! Image downloaded.';
        resultDiv.classList.remove('text-red-600');
        resultDiv.classList.add('text-green-600');
    } catch (err) {
        resultDiv.textContent = `Error: ${err.message}`;
        resultDiv.classList.add('text-red-600');
    } finally {
        progress.style.display = 'none';
        progressText.textContent = '';
        form.querySelector('button').disabled = false;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initSliders();
    updateFileSize();
    const deletePagesType = document.getElementById('deletePages-type');
    if (deletePagesType) {
        deletePagesType.addEventListener('change', toggleDeleteInputs);
    }

// page reload
    const lastTool = localStorage.getItem('lastTool');
        if (lastTool) showTool(lastTool); // Restore last tool
        else showTool('chat');

// 

    const imageToPdfForm = document.getElementById('imageToPdfForm');
    if (imageToPdfForm) {
        const imageToPdfFile = document.getElementById('imageToPdf-file');
        const descriptionPosition = document.getElementById('description-position');
        if (imageToPdfFile) {
            imageToPdfFile.addEventListener('change', function() {
                const fileName = this.files[0] ? this.files[0].name : 'No file selected';
                const fileNameElement = document.getElementById('imageToPdf-file-name');
                if (fileNameElement) {
                    fileNameElement.textContent = fileName;
                } else {
                    console.warn('Element with ID "imageToPdf-file-name" not found');
                }
            });
        } else {
            console.warn('Element with ID "imageToPdf-file" not found');
        }

        if (descriptionPosition) {
            descriptionPosition.addEventListener('change', function() {
                const customContainer = document.getElementById('custom-position-container');
                const customX = document.getElementById('custom-x');
                const customY = document.getElementById('custom-y');
                if (customContainer) {
                    const isCustom = this.value === 'custom';
                    customContainer.classList.toggle('hidden', !isCustom);
                    if (!isCustom && customX && customY) {
                        customX.value = '';
                        customY.value = '';
                    }
                } else {
                    console.warn('Element with ID "custom-position-container" not found');
                }
            });
            // Trigger change event on load
            descriptionPosition.dispatchEvent(new Event('change'));
        } else {
            console.warn('Element with ID "description-position" not found');
        }
    } else {
        console.warn('Form with ID "imageToPdfForm" not found');
    }
        
    
});