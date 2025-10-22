
async function convertPDFToImagesClientSide() {
    console.log('Starting client-side PDF to images conversion...');

    const form = document.getElementById('pdfToImagesForm');
    const fileInput = document.getElementById('pdfToImages-file');
    const resultDiv = document.getElementById('result-pdfToImagesForm');
    const progressDiv = document.getElementById('progress-pdfToImagesForm');
    const progressText = document.getElementById('progress-text-pdfToImagesForm');
    const submitButton = form.querySelector('button');

    // imagetopdf
    // Validation
    if (!fileInput || !fileInput.files[0]) {
        alert('Please select a PDF file.');
        return;
    }

    const file = fileInput.files[0];

    // Validate file type
    if (file.type !== 'application/pdf') {
        alert('Please select a PDF file.');
        return;
    }

    if (file.size > 200 * 1024 * 1024) { alert("Use PDF less than 200 mb"); return; }
    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Loading PDF...';
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-file-image mr-2"></i> Converting...';

    try {

        // // LAZY LOADING: Load required libraries. lib required w=nly when needed.
        const [pdfjs, jszip] = await pdfLibraryManager.loadLibraries([
            'pdfjs', 'jszip'
        ]);

        // Load PDF document
        const pdfBytes = await file.arrayBuffer();
        const pdf = await pdfjs.getDocument({ data: pdfBytes }).promise;
        const numPages = pdf.numPages;

        console.log(`Processing ${numPages} pages...`);

        const zip = new jszip();
        let processedPages = 0;

        for (let pageNum = 1; pageNum <= numPages; pageNum++) {
            const progress = Math.round((pageNum / numPages) * 100);
            progressText.textContent = `Converting page ${pageNum}/${numPages}... (${progress}%)`;

            const page = await pdf.getPage(pageNum);
            const viewport = page.getViewport({ scale: 2.0 }); // Higher scale for better quality

            const canvas = document.createElement('canvas');
            canvas.width = viewport.width;
            canvas.height = viewport.height;

            // const context = canvas.getContext('2d');
            const context = canvas.getContext('2d', { willReadFrequently: true });

            // White background
            context.fillStyle = 'white';
            context.fillRect(0, 0, canvas.width, canvas.height);

            // Render PDF page
            const renderContext = {
                canvasContext: context,
                viewport: viewport
            };

            await page.render(renderContext).promise;

            // Convert to blob and add to zip
            const blob = await new Promise(resolve => {
                canvas.toBlob(resolve, 'image/png', 0.9);
            });

            zip.file(`page_${pageNum}.png`, blob);
            processedPages++;

            // Clean up
            canvas.remove();
        }

        progressText.textContent = 'Creating ZIP file...';

        // Generate ZIP file
        const zipBlob = await zip.generateAsync({ type: 'blob' });

        // Download the ZIP file
        const filename = `pdf_images_${Date.now()}.zip`;
        saveAs(zipBlob, filename);

        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>PDF Converted to Images Successfully!</strong><br>
                📁 File: ${filename}<br>
                📄 Pages converted: ${processedPages}<br>
            
            </div>
        `;

    } catch (error) {
        console.error('PDF to images conversion failed:', error);
        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Conversion failed: ${error.message}<br>
                <small>Falling back to server processing...</small>
            </div>
        `;

        // Fallback to server processing
        // setTimeout(() => {
        //     processPDF('convert_pdf_to_images', 'pdfToImagesForm');
        // }, 2000);

    } finally {
        progressDiv.style.display = 'none';
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-file-image mr-2"></i> Convert to Images';
    }
}

async function splitPDFClientSide() {
    console.log('Starting client-side PDF split...');
    const [pdfLib, jszip] = await pdfLibraryManager.loadLibraries([
        'pdfLib', 'jszip'
    ]);


    const form = document.getElementById('splitForm');
    const fileInput = document.getElementById('split-file');
    const resultDiv = document.getElementById('result-splitForm');
    const progressDiv = document.getElementById('progress-splitForm');
    const progressText = document.getElementById('progress-text-splitForm');
    const submitButton = form.querySelector('button');

    // Validation
    if (!fileInput || !fileInput.files[0]) {
        alert('Please select a PDF file.');
        return;
    }

    const file = fileInput.files[0];

    // Validate file type
    if (file.type !== 'application/pdf') {
        alert('Please select a PDF file.');
        return;
    }
    if (file.size > 200 * 1024 * 1024) { alert("Use PDF less than 200 mb"); return; }
    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Loading PDF...';
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-cut mr-2"></i> Splitting...';

    try {
        // Load PDF document
        const pdfBytes = await file.arrayBuffer();
        const pdfDoc = await pdfLib.PDFDocument.load(pdfBytes);
        const pages = pdfDoc.getPages();
        const numPages = pages.length;

        console.log(`Splitting ${numPages} pages...`);

        const zip = new jszip();
        let processedPages = 0;

        for (let i = 0; i < numPages; i++) {
            const progress = Math.round(((i + 1) / numPages) * 100);
            progressText.textContent = `Splitting page ${i + 1}/${numPages}... (${progress}%)`;

            // Create new PDF with single page
            const singlePagePdf = await pdfLib.PDFDocument.create();
            const [copiedPage] = await singlePagePdf.copyPages(pdfDoc, [i]);
            singlePagePdf.addPage(copiedPage);

            // Save single page PDF
            const singlePageBytes = await singlePagePdf.save();
            zip.file(`page_${i + 1}.pdf`, singlePageBytes);
            processedPages++;
        }

        progressText.textContent = 'Creating ZIP file...';

        // Generate ZIP file
        const zipBlob = await zip.generateAsync({ type: 'blob' });

        // Download the ZIP file
        const filename = `split_pages_${Date.now()}.zip`;
        saveAs(zipBlob, filename);

        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>PDF Split Successfully!</strong><br>
                📁 File: ${filename}<br>
                📄 Pages split: ${processedPages}<br>
               
            </div>
        `;

    } catch (error) {
        console.error('PDF split failed:', error);
        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Split failed: ${error.message}<br>
                <small>Falling back to server processing...</small>
            </div>
        `;

        // Fallback to server processing
        // setTimeout(() => {
        //     processPDF('split_pdf', 'splitForm');
        // }, 2000);

    } finally {
        progressDiv.style.display = 'none';
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-cut mr-2"></i> Split PDF';
    }
}


//  PDF TO PPT

async function convertPDFToPPTClientSide() {
    console.log('Starting client-side PDF to PPT conversion...');
    const [pdfjs, pptxgen, pdfLib] = await pdfLibraryManager.loadLibraries([
        'pdfjs', 'pptxgen', 'pdfLib'
    ]);


    const form = document.getElementById('pdfToPptForm');
    const fileInput = document.getElementById('pdfToPpt-file');
    const resultDiv = document.getElementById('result-pdfToPptForm');
    const progressDiv = document.getElementById('progress-pdfToPptForm');
    const progressText = document.getElementById('progress-text-pdfToPptForm');
    const submitButton = form.querySelector('button[type="button"]');
    const conversionType = form.querySelector('input[name="conversionType"]:checked').value;

    // Validation
    if (!fileInput || !fileInput.files || !fileInput.files[0]) {
        alert('Please select a PDF file.');
        return;
    }

    const file = fileInput.files[0];

    // Validate file type
    if (file.type !== 'application/pdf') {
        alert('Please select a PDF file.');
        return;
    }
    if (file.size > 200 * 1024 * 1024) {
        alert("Use PDF less than 200 mb");
        return;
    }

    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Starting conversion...';
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-file-powerpoint mr-2"></i> Converting...';

    try {
        progressText.textContent = 'Loading PDF...';

        const pdfBytes = await file.arrayBuffer();
        const pdfDoc = await pdfLib.PDFDocument.load(pdfBytes);



        const numPages = pdfDoc.getPageCount();

        console.log(`Processing ${numPages} pages for PPT conversion...`);
        if (conversionType === 'image') {
            progressText.textContent = 'Converting pages to images...';

            const pptx = new pptxgen();
            pptx.layout = 'LAYOUT_WIDE';

            const pdfjsDoc = await pdfjs.getDocument({ data: pdfBytes }).promise;

            for (let i = 0; i < numPages; i++) {
                const progress = Math.round((i / numPages) * 80);
                progressText.textContent = `Processing page ${i + 1}/${numPages}... (${progress}%)`;

                const page = pdfDoc.getPage(i);
                const { width, height } = page.getSize();

                // Adaptive scaling based on page size
                const maxDimension = width > 2000 || height > 2000 ? 800 : 1200;
                const scale = Math.min(maxDimension / Math.max(width, height), 2);

                const canvas = document.createElement('canvas');
                canvas.width = Math.floor(width * scale);
                canvas.height = Math.floor(height * scale);

                // const context = canvas.getContext('2d');
                const context = canvas.getContext('2d', { willReadFrequently: true });
                context.fillStyle = 'white';
                context.fillRect(0, 0, canvas.width, canvas.height);

                const pdfjsPage = await pdfjsDoc.getPage(i + 1);
                const viewport = pdfjsPage.getViewport({ scale: scale });

                await pdfjsPage.render({
                    canvasContext: context,
                    viewport: viewport
                }).promise;

                // Adaptive quality based on content
                const hasComplexGraphics = canvas.width > 1000 || canvas.height > 1000;
                const quality = hasComplexGraphics ? 0.7 : 0.8;
                const format = hasComplexGraphics ? 'image/jpeg' : 'image/png';

                const imageData = canvas.toDataURL(format, quality);

                const slide = pptx.addSlide();

                // Fit to slide with proper margins
                const aspectRatio = width / height;
                // const maxWidth = 9.5;
                // const maxHeight = 6.5;
                // const maxWidth = 9.8;  // Increased from 9.5
                // const maxHeight = 7.0; // Increased from 6.5
                const maxWidth = 9.9;  // Almost full width
                const maxHeight = 7.4; // Almost full height

                let imgWidth, imgHeight;
                if (aspectRatio > maxWidth / maxHeight) {
                    imgWidth = maxWidth;
                    imgHeight = maxWidth / aspectRatio;
                } else {
                    imgHeight = maxHeight;
                    imgWidth = maxHeight * aspectRatio;
                }

                const xOffset = (10 - imgWidth) / 2 + 1.4;
                const yOffset = (7.5 - imgHeight) / 2;

                slide.addImage({
                    data: imageData,
                    x: xOffset,
                    y: yOffset,
                    w: imgWidth,
                    h: imgHeight,
                    sizing: { type: 'contain' }
                });

                canvas.remove();
            }

            progressText.textContent = 'Generating PowerPoint file...';
            await pptx.writeFile({
                fileName: `converted_${file.name.replace('.pdf', '')}.pptx`
            });
        }

        else {
            // Editable text conversion (basic implementation)
            throw new Error('Editable conversion requires other processing..');
        }

        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>PDF to PowerPoint Conversion Successful!</strong><br>
                📊 Converted ${numPages} pages to PowerPoint<br>
            </div>
        `;

    } catch (error) {
        console.error('PDF to PPT conversion failed:', error);

        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Conversion failed: ${error.message}<br>
                <small>Falling back to other processing...</small>
            </div>
        `;

        // Fallback to server processing
        // setTimeout(() => {
        //     processPDF('convert_pdf_to_ppt', 'pdfToPptForm');
        // }, 2000);

    } finally {
        progressDiv.style.display = 'none';
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-file-powerpoint mr-2"></i> Convert to PowerPoint';
    }
}


// IMAGE TO PDF


// Multiple Images to PDF function
async function convertMultipleImagesToPDFClientSide() {
    console.log('Starting client-side Multiple Images to PDF conversion...');

    const fileInput = document.getElementById('multipleImageToPdf-files');
    const resultDiv = document.getElementById('result-multipleImageToPdfForm');
    const progressDiv = document.getElementById('progress-multipleImageToPdfForm');
    const progressText = document.getElementById('progress-text-multipleImageToPdfForm');
    const submitButton = document.getElementById('imagetopdf');

    // Get form values
    const description = document.getElementById('multiple-image-description')?.value || '';
    const descriptionPosition = document.getElementById('multiple-description-position')?.value || 'bottom-center';
    const fontSize = parseInt(document.getElementById('multiple-description-font-size')?.value) || 20;
    const pageSize = document.getElementById('multiple-page_size')?.value || 'A4';
    const orientation = document.getElementById('multiple-orientation')?.value || 'Portrait';
    const fontColor = document.getElementById('multiple-font-color')?.value || '#000000';
    const fontFamily = document.getElementById('multiple-font-family')?.value || 'helvetica';
    const fontWeight = document.getElementById('multiple-font-weight')?.value || 'normal';
    const imagesPerPage = parseInt(document.getElementById('image-per-page')?.value) || 1;

    // Validation
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        alert('Please select at least one image file.');
        return;
    }

    const files = Array.from(fileInput.files);
    if (files.length > 50) { alert("Too many images.Use less than 50"); return; }

    // Validate file types and sizes
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    for (const file of files) {
        if (!allowedTypes.includes(file.type)) {
            alert(`Please select only PNG or JPEG image files. Invalid file: ${file.name}`);
            return;
        }
        if (file.size > 20 * 1024 * 1024) {
            alert(`File ${file.name} is too large. Maximum size is 20MB.`);
            return;
        }
    }

    // Total size validation
    let totalSize = 0;
    for (const file of files) {
        totalSize += file.size;
    }
    const totalSizeMB = totalSize / (1024 * 1024);
    if (totalSizeMB > 200) { // 500MB total limit
        alert(`Total file size (${totalSizeMB.toFixed(2)}MB) exceeds 200MB limit. Please select smaller files or fewer images.`);
        return;
    }
    const imageOrderInput = document.getElementById('multipleImageToPdf-image-order');
    let orderedFiles = files;

    if (imageOrderInput && imageOrderInput.value) {
        const order = imageOrderInput.value.split(',').map(Number);

        // Create a new array with files in the correct order
        orderedFiles = order.map(index => files[index]).filter(file => file !== undefined);

        console.log('Original order:', files.map(f => f.name));
        console.log('Reordered files:', orderedFiles.map(f => f.name));
    }


    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Starting conversion...';
    if (submitButton) {
        submitButton.disabled = true;
        submitButton.innerHTML = '<i class="fas fa-file-pdf mr-2"></i> Converting...';
    }

    try {
        progressText.textContent = 'Loading PDF library...';

        // Load PDF library
        // const [PDFLib] = await pdfLibraryManager.loadLibraries(['pdfLib']);
        const [pdfLib] = await pdfLibraryManager.loadLibraries([
            'pdfLib'
        ]);

        const { PDFDocument, rgb } = pdfLib;

        // Create new PDF document
        const pdfDoc = await PDFDocument.create();

        // Set page size
        const pageDimensions = getPageDimensions(pageSize, orientation);

        // Convert hex color to RGB
        const hexToRgb = (hex) => {
            // Remove # if present
            hex = hex.replace(/^#/, '');

            // Parse hex values
            let r, g, b;
            if (hex.length === 3) {
                r = parseInt(hex[0] + hex[0], 16);
                g = parseInt(hex[1] + hex[1], 16);
                b = parseInt(hex[2] + hex[2], 16);
            } else if (hex.length === 6) {
                r = parseInt(hex.substring(0, 2), 16);
                g = parseInt(hex.substring(2, 4), 16);
                b = parseInt(hex.substring(4, 6), 16);
            } else {
                // Default to black if invalid
                return { r: 0, g: 0, b: 0 };
            }

            return {
                r: r / 255,
                g: g / 255,
                b: b / 255
            };
        };

        const color = hexToRgb(fontColor);

        // Get the appropriate font
        let font;
        switch (fontFamily) {
            case 'times':
                font = pdfDoc.embedStandardFont(pdfLib.StandardFonts.TimesRoman);
                break;
            case 'courier':
                font = pdfDoc.embedStandardFont(pdfLib.StandardFonts.Courier);
                break;
            case 'zapf':
                font = pdfDoc.embedStandardFont(pdfLib.StandardFonts.ZapfDingbats);
                break;
            case 'helvetica':
            default:
                font = fontWeight === 'bold'
                    ? pdfDoc.embedStandardFont(pdfLib.StandardFonts.HelveticaBold)
                    : pdfDoc.embedStandardFont(pdfLib.StandardFonts.Helvetica);
                break;
        }

        let currentPage = null;
        let imagesOnCurrentPage = 0;
        const totalImages = files.length;

        for (let i = 0; i < totalImages; i++) {
            const file = orderedFiles[i];
            const progress = Math.round((i / totalImages) * 90);
            progressText.textContent = `Processing image ${i + 1} of ${totalImages}... (${progress}%)`;

            // Create new page if needed
            if (currentPage === null || imagesOnCurrentPage >= imagesPerPage) {
                currentPage = pdfDoc.addPage([pageDimensions.width, pageDimensions.height]);
                imagesOnCurrentPage = 0;
            }

            // Load image
            let image;
            try {
                if (file.type === 'image/png') {
                    image = await pdfDoc.embedPng(await file.arrayBuffer());
                } else {
                    image = await pdfDoc.embedJpg(await file.arrayBuffer());
                }
            } catch (imageError) {
                console.error(`Error loading image ${file.name}:`, imageError);
                progressText.textContent = `Skipping invalid image: ${file.name}`;
                continue; // Skip to next image
            }

            // Calculate layout based on images per page
            const imagesPerRow = imagesPerPage === 1 ? 1 : 2;
            const rows = Math.ceil(imagesPerPage / imagesPerRow);

            const margin = 50;
            const horizontalSpacing = 20;
            const verticalSpacing = description ? 60 : 20;

            const availableWidth = pageDimensions.width - (2 * margin) - ((imagesPerRow - 1) * horizontalSpacing);
            const availableHeight = pageDimensions.height - (2 * margin) - ((rows - 1) * verticalSpacing);

            const imageWidth = availableWidth / imagesPerRow;
            const imageHeight = availableHeight / rows;

            // Calculate position for current image
            const rowIndex = Math.floor(imagesOnCurrentPage / imagesPerRow);
            const colIndex = imagesOnCurrentPage % imagesPerRow;

            const x = margin + (colIndex * (imageWidth + horizontalSpacing));
            const y = pageDimensions.height - margin - ((rowIndex + 1) * imageHeight) + (rowIndex * verticalSpacing);

            // Scale image to fit the allocated space
            const imageDims = image.scaleToFit(imageWidth, imageHeight);
            const centeredX = x + (imageWidth - imageDims.width) / 2;
            const centeredY = y + (imageHeight - imageDims.height) / 2;

            // Draw image
            currentPage.drawImage(image, {
                x: centeredX,
                y: centeredY,
                width: imageDims.width,
                height: imageDims.height,
            });

            // Add description if provided
            if (description.trim()) {
                const textWidth = font.widthOfTextAtSize(description, fontSize);
                const textMargin = 10;

                let textX, textY;

                switch (descriptionPosition) {
                    case 'top':
                        textX = x + (imageWidth - textWidth) / 2;
                        textY = y + imageHeight - textMargin;
                        break;
                    case 'top-center':
                        textX = x + (imageWidth - textWidth) / 2;
                        textY = y + imageHeight - textMargin - fontSize;
                        break;
                    case 'top-left':
                        textX = x + textMargin;
                        textY = y + imageHeight - textMargin;
                        break;
                    case 'top-right':
                        textX = x + imageWidth - textWidth - textMargin;
                        textY = y + imageHeight - textMargin;
                        break;
                    case 'bottom':
                        textX = x + (imageWidth - textWidth) / 2;
                        textY = y + textMargin + fontSize;
                        break;
                    case 'bottom-center':
                        textX = x + (imageWidth - textWidth) / 2;
                        textY = y + textMargin;
                        break;
                    case 'bottom-left':
                        textX = x + textMargin;
                        textY = y + textMargin + fontSize;
                        break;
                    case 'bottom-right':
                        textX = x + imageWidth - textWidth - textMargin;
                        textY = y + textMargin + fontSize;
                        break;
                    default:
                        textX = x + (imageWidth - textWidth) / 2;
                        textY = y + textMargin;
                        break;
                }

                // Ensure text stays within bounds
                textX = Math.max(x + textMargin, Math.min(textX, x + imageWidth - textWidth - textMargin));
                textY = Math.max(y + textMargin, Math.min(textY, y + imageHeight - textMargin));

                currentPage.drawText(description, {
                    x: textX,
                    y: textY,
                    size: fontSize,
                    color: rgb(color.r, color.g, color.b),
                    font: font,
                    maxWidth: imageWidth - (2 * textMargin),
                });
            }

            imagesOnCurrentPage++;
        }

        progressText.textContent = 'Generating PDF...';

        // Save PDF
        const pdfBytes = await pdfDoc.save();
        const pdfBlob = new Blob([pdfBytes], { type: 'application/pdf' });

        // Download
        const url = URL.createObjectURL(pdfBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `converted_images_${Date.now()}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>Images to PDF Conversion Successful!</strong><br>
                🖼️ Converted ${orderedFiles.length} images to PDF<br>
                📍 Description Position: ${descriptionPosition}<br>
                📄 Images per Page: ${imagesPerPage}
            </div>
        `;

    } catch (error) {
        console.error('Multiple Images to PDF conversion failed:', error);
        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Conversion failed: ${error.message}<br>
                <small>Please try again or use smaller images.</small>
            </div>
        `;
    } finally {
        progressDiv.style.display = 'none';
        progressText.textContent = '';
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.innerHTML = 'Multiple Images to PDF';
        }
    }
}

// Page dimensions helper function
function getPageDimensions(pageSize, orientation) {
    const sizes = {
        'A4': { width: 595.28, height: 841.89 }, // A4 in points (72 DPI)
        'Letter': { width: 612, height: 792 }     // Letter in points
    };

    const size = sizes[pageSize] || sizes['A4'];
    return orientation === 'Landscape'
        ? { width: size.height, height: size.width }
        : size;
}

// Image orientation function (simplified)
function loadAndFixImageOrientation(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        img.onload = function () {
            try {
                canvas.width = img.width;
                canvas.height = img.height;

                // Draw image without orientation fixes (basic version)
                ctx.drawImage(img, 0, 0, img.width, img.height);

                // Convert to blob
                canvas.toBlob(blob => {
                    if (blob) {
                        blob.arrayBuffer().then(resolve).catch(reject);
                    } else {
                        reject(new Error('Canvas to blob conversion failed'));
                    }
                }, file.type || 'image/jpeg', 0.9);
            } catch (error) {
                reject(error);
            }
        };

        img.onerror = function () {
            reject(new Error('Failed to load image'));
        };

        img.src = URL.createObjectURL(file);
    });
}

// // Initialize image previews
function initializeMultipleImagePreview() {
    const fileInput = document.getElementById('multipleImageToPdf-files');
    const filesCount = document.getElementById('multipleImageToPdf-files-count');
    const imagePreviews = document.getElementById('image-previews');
    const imageList = document.getElementById('image-list');

    if (!fileInput || !filesCount || !imagePreviews || !imageList) {
        console.warn('Image preview elements not found');
        return;
    }

    fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);

        if (files.length === 0) {
            imagePreviews.classList.add('hidden');
            filesCount.textContent = 'No files selected';
            return;
        }

        filesCount.textContent = `${files.length} file(s) selected`;
        imageList.innerHTML = '';
        imagePreviews.classList.remove('hidden');

        files.forEach((file, index) => {
            // Validate file type
            if (!file.type.startsWith('image/')) {
                console.warn(`Skipping non-image file: ${file.name}`);
                return;
            }

            const reader = new FileReader();

            reader.onload = (e) => {
                const imageDiv = document.createElement('div');
                imageDiv.className = 'image-preview border border-gray-200 rounded-lg p-3 bg-white';
                imageDiv.dataset.fileIndex = index;

                imageDiv.innerHTML = `
                    <div class="flex flex-col items-center">
                        <img src="${e.target.result}" alt="${file.name}" 
                             class="mb-2 max-h-32 max-w-full object-contain border border-gray-300 rounded">
                        <div class="flex items-center justify-between w-full mt-2">
                            <span class="text-gray-600 text-xs truncate flex-1 mr-2" 
                                  title="${file.name}">${file.name}</span>
                            <div class="flex space-x-1">
                                <button type="button" class="move-up bg-blue-600 text-white px-2 py-1 rounded text-xs hover:bg-blue-700 transition-colors">
                                    <i class="fas fa-arrow-up"></i>
                                </button>
                                <button type="button" class="move-down bg-blue-600 text-white px-2 py-1 rounded text-xs hover:bg-blue-700 transition-colors">
                                    <i class="fas fa-arrow-down"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                `;

                imageList.appendChild(imageDiv);

                // Add event listeners for move buttons
                const moveUp = imageDiv.querySelector('.move-up');
                const moveDown = imageDiv.querySelector('.move-down');

                if (moveUp) {
                    moveUp.addEventListener('click', (event) => {
                        event.preventDefault();
                        const prev = imageDiv.previousElementSibling;
                        if (prev) {
                            imageList.insertBefore(imageDiv, prev);
                            updateImageOrder();
                        }
                    });
                }

                if (moveDown) {
                    moveDown.addEventListener('click', (event) => {
                        event.preventDefault();
                        const next = imageDiv.nextElementSibling;
                        if (next) {
                            imageList.insertBefore(next, imageDiv);
                            updateImageOrder();
                        }
                    });
                }
            };

            reader.onerror = () => {
                console.error(`Failed to read file: ${file.name}`);
            };

            reader.readAsDataURL(file);
        });

        updateImageOrder();
    });
}

// // Update image order function
function updateImageOrder() {
    const imagePreviews = document.querySelectorAll('.image-preview');
    const imageOrder = Array.from(imagePreviews).map(preview => preview.dataset.fileIndex);
    const imageOrderInput = document.getElementById('multipleImageToPdf-image-order');

    if (imageOrderInput) {
        imageOrderInput.value = imageOrder.join(',');
        console.log("Updated image order:", imageOrderInput.value);
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function () {
    initializeMultipleImagePreview();

    // Add file label update if function exists
    if (typeof updateFileLabel === 'function') {
        const fileInput = document.getElementById('multipleImageToPdf-files');
        const fileCount = document.getElementById('multipleImageToPdf-files-count');
        if (fileInput && fileCount) {
            updateFileLabel('multipleImageToPdf-files', 'multipleImageToPdf-files-count');
        }
    }
});





//  ADD PAGE NUMBER

async function addPageNumbersClientSide() {
    console.log('Starting client-side page numbering...');
    // const [pdfLib] = await pdfLibraryManager.loadLibraries([
    //     'pdfLib'
    // ]);
    const [pdfLib] = await pdfLibraryManager.loadLibraries([
        'pdfLib'
    ]);

    const form = document.getElementById('pageNumbersForm');
    const fileInput = document.getElementById('pageNumbers-file');
    const positionSelect = document.getElementById('pageNumbers-position');
    const alignmentSelect = document.getElementById('pageNumbers-alignment');
    const formatSelect = document.getElementById('pageNumbers-format');
    const resultDiv = document.getElementById('result-pageNumbersForm');
    const progressDiv = document.getElementById('progress-pageNumbersForm');
    const progressText = document.getElementById('progress-text-pageNumbersForm');
    const submitButton = form.querySelector('button[type="button"]');

    // Validation
    if (!fileInput || !fileInput.files || !fileInput.files[0]) {
        alert('Please select a PDF file.');
        return;
    }

    const file = fileInput.files[0];

    // Validate file type
    if (file.type !== 'application/pdf') {
        alert('Please select a PDF file.');
        return;
    }
    if (file.size > 200 * 1024 * 1024) { alert("Use PDF less than 200 mb"); return; }

    // Get form values
    const position = positionSelect ? positionSelect.value : 'bottom';
    const alignment = alignmentSelect ? alignmentSelect.value : 'center';
    const format = formatSelect ? formatSelect.value : 'page_x';

    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Starting page numbering...';
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-list-ol mr-2"></i> Adding Page Numbers...';

    try {
        progressText.textContent = 'Loading PDF...';


        if (!pdfLibraryManager.libraries.pdfLib || !pdfLibraryManager.libraries.pdfLib.loaded) {
            throw new Error('PDF library not loaded. Please ensure libraries are loaded first.');
        }


        const { PDFDocument, rgb, StandardFonts } = pdfLib;

        // Load PDF
        const pdfBytes = await file.arrayBuffer();
        const pdfDoc = await PDFDocument.load(pdfBytes);
        const numPages = pdfDoc.getPageCount();

        console.log(`Adding page numbers to ${numPages} pages...`);

        // Embed font
        const font = await pdfDoc.embedFont(StandardFonts.Helvetica);

        // Process each page
        for (let i = 0; i < numPages; i++) {
            const progress = Math.round((i / numPages) * 90);
            progressText.textContent = `Processing page ${i + 1}/${numPages}... (${progress}%)`;

            const page = pdfDoc.getPage(i);
            const { width, height } = page.getSize();

            // Generate page number text
            const pageNumber = i + 1;
            const pageText = format === 'page_x' ? `Page ${pageNumber}` : `${pageNumber}`;

            // Calculate position
            const margin = 30; // Distance from edge
            const fontSize = 12;

            let x, y;

            // Vertical position
            switch (position) {
                case 'top':
                    y = height - margin;
                    break;
                case 'bottom':
                default:
                    y = margin;
                    break;
            }

            // Horizontal alignment
            switch (alignment) {
                case 'left':
                    x = margin;
                    break;
                case 'right':
                    x = width - margin - font.widthOfTextAtSize(pageText, fontSize);
                    break;
                case 'center':
                default:
                    x = (width - font.widthOfTextAtSize(pageText, fontSize)) / 2;
                    break;
            }

            // Add page number
            page.drawText(pageText, {
                x,
                y,
                size: fontSize,
                font: font,
                color: rgb(0, 0, 0), // Black color
            });

            console.log(`Added page number "${pageText}" to page ${pageNumber} at (${x}, ${y})`);
        }

        progressText.textContent = 'Saving PDF with page numbers...';

        // Save the modified PDF
        const pdfWithNumbers = await pdfDoc.save();
        const pdfBlob = new Blob([pdfWithNumbers], { type: 'application/pdf' });

        // Download
        const url = URL.createObjectURL(pdfBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `numbered_${file.name}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>Page Numbers Added Successfully!</strong><br>
                📄 Added numbers to ${numPages} pages<br>
                📍 Position: ${position}, Alignment: ${alignment}<br>
                
            </div>
        `;

        console.log(`Successfully added page numbers to ${numPages} pages`);

    } catch (error) {
        console.error('Page numbering failed:', error);

        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Failed to add page numbers: ${error.message}<br>
                <small>Falling back to server processing...</small>
            </div>
        `;

        // Fallback to server processing
        // setTimeout(() => {
        //     processPDF('add_page_numbers', 'pageNumbersForm');
        // }, 2000);

    } finally {
        progressDiv.style.display = 'none';
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-list-ol mr-2"></i> Add Page Numbers';
    }
}




//  reorder function with validation

async function reorderPDFPagesClientSide() {
    console.log('Starting client-side PDF page reordering...');
    const [pdfLib,] = await pdfLibraryManager.loadLibraries([
        'pdfLib'
    ]);

    const form = document.getElementById('reorderForm');
    const fileInput = document.getElementById('reorder-file');
    const resultDiv = document.getElementById('result-reorderForm');
    const progressDiv = document.getElementById('progress-reorderForm');
    const progressText = document.getElementById('progress-text-reorderForm');
    const pageOrderInput = document.getElementById('reorder-page-order');
    // const submitButton = form.querySelector('button[type="button"]');
    const submitButton = document.getElementById('reorder-submit-btn');

    // Validation
    if (!fileInput || !fileInput.files || !fileInput.files.length) {
        alert('Please select a PDF file.');
        return;
    }

    const file = fileInput.files[0];

    if (file.type !== 'application/pdf') {
        alert('Please select a PDF file.');
        return;
    }

    if (file.size > 200 * 1024 * 1024) {
        alert('File size exceeds 200MB limit for processing.');
        return;
    }

    if (!pageOrderInput || !pageOrderInput.value) {
        alert('Please reorder pages before processing.');
        return;
    }

    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Starting page reordering...';
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-sort-numeric-up mr-2"></i> Reordering...';

    try {
        // Parse and validate page order
        const pageOrder = pageOrderInput.value.split(',').map(p => parseInt(p.trim()));

        console.log('Requested page order:', pageOrder);

        progressText.textContent = 'Loading PDF document...';

        // Load PDF to get actual page count
        const { PDFDocument } = pdfLib;
        const pdfBytes = await file.arrayBuffer();
        const pdfDoc = await PDFDocument.load(pdfBytes);

        const totalPages = pdfDoc.getPageCount();
        console.log('Total pages in PDF:', totalPages);

        // Validate page order
        if (pageOrder.length !== totalPages) {
            throw new Error(`Page order has ${pageOrder.length} pages but PDF has ${totalPages} pages.`);
        }

        // Check for valid page numbers and duplicates
        const seenPages = new Set();
        for (const pageNum of pageOrder) {
            if (pageNum < 1 || pageNum > totalPages) {
                throw new Error(`Invalid page number: ${pageNum}. Must be between 1 and ${totalPages}.`);
            }
            if (seenPages.has(pageNum)) {
                throw new Error(`Duplicate page number found: ${pageNum}. Each page must appear exactly once.`);
            }
            seenPages.add(pageNum);
        }

        // Create new PDF document
        const newPdfDoc = await PDFDocument.create();

        progressText.textContent = 'Reordering pages... (0%)';

        // Copy pages in the specified order
        for (let i = 0; i < pageOrder.length; i++) {
            const pageNum = pageOrder[i];
            const pageIndex = pageNum - 1;

            // Final bounds check (should never fail if validation passed)
            if (pageIndex < 0 || pageIndex >= totalPages) {
                throw new Error(`Internal error: Cannot access page ${pageNum}`);
            }

            // Safe copy
            const [copiedPage] = await newPdfDoc.copyPages(pdfDoc, [pageIndex]);
            newPdfDoc.addPage(copiedPage);
        }

        progressText.textContent = 'Finalizing PDF... (95%)';

        // Save the reordered PDF
        const reorderedPdfBytes = await newPdfDoc.save();
        const reorderedBlob = new Blob([reorderedPdfBytes], { type: 'application/pdf' });

        progressText.textContent = 'Downloading... (100%)';

        // Download the reordered PDF
        const filename = `reordered_${file.name.replace('.pdf', '')}.pdf`;
        const url = URL.createObjectURL(reorderedBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        // Show success message
        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>PDF Pages Reordered Successfully!</strong><br>
                📄 Pages reordered: ${pageOrder.join(' → ')}<br>
                <small>Original order: ${Array.from({ length: totalPages }, (_, i) => i + 1).join(', ')}</small>
            </div>
        `;

        console.log('page reordering completed successfully');

    } catch (error) {
        console.error('page reordering failed:', error);

        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Page reordering failed: ${error.message}<br>
                <small>Falling back to server processing...</small>
            </div>
        `;

        // Fallback to server processing
        // setTimeout(() => {
        //     processPDF('reorder_pages', 'reorderForm');
        // }, 2000);

    } finally {
        // Clean up
        progressDiv.style.display = 'none';
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-sort-numeric-up mr-2"></i> Reorder and Download ';
    }
}


// saves this function called in pdf image and split pdf
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



//   rotate page


// Global variables for rotation
let currentPDFDoc = null;
let currentRotationAngle = 90;
let selectedPagesForRotation = new Set();
let pageRotations = new Map(); // Store current rotation for each page


// Handle file selection
async function handleFileSelectForRotatePages() {
    console.log("1. Function called");
    const [pdfjs, pdfLib, fileSaver] = await pdfLibraryManager.loadLibraries([
        'pdfjs', 'pdfLib', 'fileSaver'
    ]);

    const fileInput = document.getElementById('rotatePages-file');
    const container = document.getElementById('rotate-pages-container');



    // if (!fileInput.files[0]) {
    //     console.log("No file selected");
    //     // alert(`⚠️ select pdf file`);
    //     return;
    // }

    // 200 MB size validation
    const maxSizeMB = 200;
    const fileSizeMB = fileInput.files[0].size / (1024 * 1024); // bytes → MB
    if (fileSizeMB > maxSizeMB) {
        alert(`⚠️ File too large! Please upload a PDF smaller than ${maxSizeMB} MB.`);
        console.warn(`File rejected: ${fileSizeMB.toFixed(2)} MB exceeds ${maxSizeMB} MB limit.`);
        fileInput.value = ""; // Clear input so user can select again
        return;
    }

    try {

        const arrayBuffer = await fileInput.files[0].arrayBuffer();
        currentPDFDoc = await pdfjs.getDocument({ data: arrayBuffer }).promise;

        await loadPagePreviewsForRotation();

    } catch (error) {
        console.log("10. ERROR:", error);
    }
}

// Load page previews
async function loadPagePreviewsForRotation() {
    // console.log("10. Loading page previews...");
    const container = document.getElementById('rotate-pages-preview-container');
    const numPages = currentPDFDoc.numPages;
    document.getElementById('vishnuji').style.display = 'flex';
    document.getElementById('infomessage').style.display = 'none';

    

    // Show loading
    container.innerHTML = '<div class="col-span-3 text-center py-8"><div class="animate-spin rounded-full h-12 w-12 border-b-2 border-amber-600 mx-auto flex justify-center"></div><p class="text-gray-600 mt-2">Loading pages...</p></div>';

    let html = '';

    for (let i = 1; i <= numPages; i++) {
        const page = await currentPDFDoc.getPage(i);
        const viewport = page.getViewport({ scale: 0.4 });

        const canvas = document.createElement('canvas');
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        const context = canvas.getContext('2d');

        // White background
        context.fillStyle = 'white';
        context.fillRect(0, 0, canvas.width, canvas.height);

        await page.render({ canvasContext: context, viewport }).promise;

        const canvasDataUrl = canvas.toDataURL();
        const currentRotation = pageRotations.get(i) || 0;

        html += `
        <div class="page-rotate-item border-2 border-gray-200 rounded-lg p-3 bg-white cursor-pointer mb-4" 
             data-page="${i}" 
             onclick="togglePageSelection(${i})">
            <div class="flex flex-col items-center">
                <!-- Page Preview Image -->
                <div class=" mt-3">
                    <img src="${canvasDataUrl}" alt="Page ${i}" 
                         class="border border-gray-300 rounded max-w-full h-auto"
                         style="transform: rotate(${currentRotation}deg); transition: transform 0.3s ease;">
                </div>
                
                <!-- Rotation buttons for this page - SAME COLORS AS EXISTING BUTTONS -->
                <div class="rotation-buttons flex gap-2 mb-3 w-full justify-center" style="display: flex !important; visibility: visible !important;">
                    <button type="button" onclick="event.stopPropagation(); rotateSinglePage(${i}, 90)" 
                            class="bg-blue-100 border-2 border-blue-300 rounded-lg p-2 hover:bg-blue-200 transition-colors text-blue-700 font-medium" 
                            style="display: inline-block !important; visibility: visible !important;">
                        <i class="fas fa-redo text-blue-600 text-sm mb-1"></i>
                        <div class="text-xs">90° Right</div>
                    </button>
                    <button type="button" onclick="event.stopPropagation(); rotateSinglePage(${i}, -90)" 
                            class="bg-green-100 border-2 border-green-300 rounded-lg p-2 hover:bg-green-200 transition-colors text-green-700 font-medium"
                            style="display: inline-block !important; visibility: visible !important;">
                        <i class="fas fa-undo text-green-600 text-sm mb-1"></i>
                        <div class="text-xs">90° Left</div>
                    </button>
                    <button type="button" onclick="event.stopPropagation(); rotateSinglePage(${i}, 180)" 
                            class="bg-purple-100 border-2 border-purple-300 rounded-lg p-2 hover:bg-purple-200 transition-colors text-purple-700 font-medium"
                            style="display: inline-block !important; visibility: visible !important;">
                        <i class="fas fa-sync-alt text-purple-600 text-sm mb-1"></i>
                        <div class="text-xs">180°</div>
                    </button>
                </div>
                
                <!-- Page info -->
                <div class="flex items-center justify-between w-full">
                    <span class="text-gray-700 font-medium">Page ${i}</span>
                    <div class="text-xs ${currentRotation !== 0 ? 'text-amber-600 font-bold' : 'text-gray-500'}">
                        ${currentRotation !== 0 ? `${currentRotation}°` : 'Original'}
                    </div>
                </div>
                
                <!-- Selection indicator -->
                <div class="selection-indicator text-xs mt-2 ${selectedPagesForRotation.has(i) ? 'text-red-600 font-bold' : 'hidden'}">
                    ${selectedPagesForRotation.has(i) ? 'SELECTED' : ''}
                </div>
            </div>
        </div>
        `;
        canvas.remove();
    }

    container.innerHTML = html;
    console.log("13. Page previews loaded with rotation buttons");
}
// Toggle page selection
function togglePageSelection(pageNum) {
    if (selectedPagesForRotation.has(pageNum)) {
        selectedPagesForRotation.delete(pageNum);
    } else {
        selectedPagesForRotation.add(pageNum);
    }

    // Update UI
    const pageElement = document.querySelector(`[data-page="${pageNum}"]`);
    if (pageElement) {
        if (selectedPagesForRotation.has(pageNum)) {
            pageElement.classList.remove('border-gray-200');
            pageElement.classList.add('border-red-500', 'bg-red-50');
        } else {
            pageElement.classList.remove('border-red-500', 'bg-red-50');
            pageElement.classList.add('border-gray-200');
        }

        // Update selection indicator
        const selectionIndicator = pageElement.querySelector('.text-xs:last-child');
        if (selectionIndicator) {
            if (selectedPagesForRotation.has(pageNum)) {
                selectionIndicator.textContent = 'SELECTED';
                selectionIndicator.classList.remove('hidden');
                selectionIndicator.classList.add('text-red-600', 'font-medium');
            } else {
                selectionIndicator.classList.add('hidden');
            }
        }
    }
}
// Set rotation angle with red background
function setRotationAngle(angle) {
    currentRotationAngle = angle;

    // Remove red background from all buttons
    document.querySelectorAll('.rotate-angle-btn').forEach(btn => {
        btn.classList.remove('bg-red-300', 'border-red-500');
        btn.classList.add('bg-blue-100', 'border-blue-300', 'bg-green-100', 'border-green-300', 'bg-purple-100', 'border-purple-300');
    });

    // Add red background to selected button
    const selectedBtn = event.currentTarget;
    selectedBtn.classList.remove('bg-blue-100', 'border-blue-300', 'bg-green-100', 'border-green-300', 'bg-purple-100', 'border-purple-300');
    selectedBtn.classList.add('bg-red-300', 'border-red-500');
}

// Rotate selected pages with immediate visual feedback
function rotateSelectedPages() {
    if (selectedPagesForRotation.size === 0) {
        alert('Please select pages to rotate by clicking on them.');
        return;
    }

    const resultDiv = document.getElementById('result-rotatePagesForm');

    // Apply rotation to selected pages
    for (const pageNum of selectedPagesForRotation) {
        const currentRotation = pageRotations.get(pageNum) || 0;
        const newRotation = (currentRotation + currentRotationAngle) % 360;
        pageRotations.set(pageNum, newRotation);

        // Immediate visual update
        const pageElement = document.querySelector(`[data-page="${pageNum}"]`);
        if (pageElement) {
            const img = pageElement.querySelector('img');
            img.style.transform = `rotate(${newRotation}deg)`;

            // Update rotation text
            const rotationText = pageElement.querySelector('.text-xs:nth-child(2)');
            if (rotationText) {
                rotationText.textContent = `${newRotation}°`;
                rotationText.className = 'text-xs text-amber-600 font-bold';
            }
        }
    }

    // Show success message
    resultDiv.innerHTML = `
        <div class="text-green-600 text-center">
            ✅ Rotated ${selectedPagesForRotation.size} page(s) by ${currentRotationAngle}°
        </div>
    `;

    // Clear selection after rotation
    // selectedPagesForRotation.clear();

    // Update all page borders to remove selection
    // document.querySelectorAll('.page-rotate-item').forEach(item => {
    //     item.classList.remove('border-red-500', 'bg-red-50');
    //     item.classList.add('border-gray-200');

    //     // Hide selection indicator
    //     const selectionIndicator = item.querySelector('.text-xs:last-child');
    //     if (selectionIndicator) {
    //         selectionIndicator.classList.add('hidden');
    //     }
    // });

    // Auto-hide message
    setTimeout(() => {
        resultDiv.innerHTML = '';
    }, 3000);
    // selectedPagesForRotation.clear();
    // toggleSelectAllPages()
    // const button = document.getElementById('myid');
    // button.classList.remove('bg-green-600', 'hover:bg-green-700');
    // button.classList.add('bg-gray-600', 'hover:bg-gray-700');
    // button.innerHTML = '<i class="fas fa-check-square mr-2"></i> Select All';

    // button.style.backgroundColor = '#4B5563';
}

// Download final rotated PDF
async function downloadRotatedPDF() {
    if (!currentPDFDoc) {
        alert('Please upload a PDF file first.');
        return;
    }

    const progressDiv = document.getElementById('progress-rotatePagesForm');
    const progressText = document.getElementById('progress-text-rotatePagesForm');
    const resultDiv = document.getElementById('result-rotatePagesForm');

    progressDiv.style.display = 'block';
    progressText.textContent = 'Creating rotated PDF...';

    try {
        const [pdfLib] = await pdfLibraryManager.loadLibraries(['pdfLib']);
        const { PDFDocument, degrees } = pdfLib;

        // Get original PDF bytes
        const originalArrayBuffer = await currentPDFDoc.getData();
        const pdfDoc = await PDFDocument.load(originalArrayBuffer);

        progressText.textContent = 'Applying rotations...';

        // Apply all rotations
        let rotatedCount = 0;
        for (const [pageNum, rotation] of pageRotations) {
            if (rotation !== 0) {
                const pageIndex = pageNum - 1;
                const page = pdfDoc.getPage(pageIndex);
                page.setRotation(degrees(rotation));
                rotatedCount++;
            }
        }

        progressText.textContent = 'Saving PDF...';

        // Save and download
        const rotatedPdfBytes = await pdfDoc.save();
        const rotatedBlob = new Blob([rotatedPdfBytes], { type: 'application/pdf' });

        const fileInput = document.getElementById('rotatePages-file');
        const originalName = fileInput.files[0].name.replace('.pdf', '');
        const filename = `rotated_${originalName}.pdf`;

        const url = URL.createObjectURL(rotatedBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        resultDiv.innerHTML = `
            <div class="text-green-600 text-center">
                ✅ PDF Downloaded Successfully!<br>
                <small>${rotatedCount} pages rotated</small>
            </div>
        `;

    } catch (error) {
        console.error('Download failed:', error);
        resultDiv.innerHTML = `
            <div class="text-red-600 text-center">
                ❌ Download failed: ${error.message}
            </div>
        `;
    } finally {
        progressDiv.style.display = 'none';
        // selectedPagesForRotation.clear();
    }
}


function toggleSelectAllPages() {
    const button = document.getElementById('myid');
    const allPages = document.querySelectorAll('.page-rotate-item');
    const allSelected = allPages.length > 0 && selectedPagesForRotation.size === allPages.length;

    if (allSelected) {
        // Deselect all - change to gray
        button.classList.remove('bg-green-600', 'hover:bg-green-700');
        button.classList.add('bg-gray-600', 'hover:bg-gray-700');
        button.innerHTML = '<i class="fas fa-check-square mr-2"></i> Select All';

        selectedPagesForRotation.clear();
        allPages.forEach(page => {
            page.classList.remove('border-red-500', 'bg-red-50');
            page.classList.add('border-gray-200');
        });
    } else {
        // Select all - change to green
        button.classList.remove('bg-gray-600', 'hover:bg-gray-700');
        button.classList.add('bg-green-600', 'hover:bg-green-700');
        button.innerHTML = '<i class="fas fa-times-circle mr-2"></i> Deselect All';

        allPages.forEach(page => {
            const pageNum = parseInt(page.dataset.page);
            selectedPagesForRotation.add(pageNum);
            page.classList.remove('border-gray-200');
            page.classList.add('border-red-500', 'bg-red-50');
        });
    }

    // updateSelectedPagesDisplay();

}
function rotateSinglePage(pageNum, angle) {
    const currentRotation = pageRotations.get(pageNum) || 0;
    const newRotation = (currentRotation + angle) % 360;
    pageRotations.set(pageNum, newRotation);

    // Immediate visual update
    const pageElement = document.querySelector(`[data-page="${pageNum}"]`);
    if (pageElement) {
        const img = pageElement.querySelector('img');
        img.style.transform = `rotate(${newRotation}deg)`;

        const rotationText = pageElement.querySelector('.text-xs');
        if (rotationText) {
            rotationText.textContent = `${newRotation}°`;
            rotationText.className = 'text-xs text-amber-600 font-bold';
        }
    }
}
// Initialize file label
updateFileLabel('rotatePages-file', 'rotatePages-file-name');

/////////////////////////////////////////////////////////////////////////////////////////////////////////


// Global variables
let mainPDFDoc = null;
let insertPDFDoc = null;
let mainPageOrder = []; // Track current page order
let selectedMainPages = new Set(); // Track selected pages for deletion
let isDragging = false;
let dragSrcEl = null;



//  helper fn
// Update DOM indices and button states without reloading
function updatePageIndicesAndButtons() {
    const pageItems = document.querySelectorAll('.page-insert-item');
    
    pageItems.forEach((item, index) => {
        // Update the data index
        item.dataset.currentIndex = index;
        
        // Get the buttons
        const upButton = item.querySelector('button[onclick*="movePageUp"]');
        const downButton = item.querySelector('button[onclick*="movePageDown"]');
        
        // Update button onclick handlers with new indices
        if (upButton) {
            upButton.onclick = (e) => {
                e.stopPropagation();
                movePageUp(index);
            };
            upButton.disabled = index === 0;
            upButton.classList.toggle('opacity-50', index === 0);
            upButton.classList.toggle('cursor-not-allowed', index === 0);
        }
        
        if (downButton) {
            downButton.onclick = (e) => {
                e.stopPropagation();
                movePageDown(index);
            };
            downButton.disabled = index === pageItems.length - 1;
            downButton.classList.toggle('opacity-50', index === pageItems.length - 1);
            downButton.classList.toggle('cursor-not-allowed', index === pageItems.length - 1);
        }
        
        // Update the page number display if needed
        const pageNumberSpan = item.querySelector('span.text-gray-700');
        if (pageNumberSpan) {
            const originalPageNum = mainPageOrder[index];
            pageNumberSpan.textContent = `Page ${originalPageNum}`;
        }
    });
}


// Update position dropdown based on current page order
function updatePositionDropdown() {
    const positionSelect = document.getElementById('insertPdf-position');
    positionSelect.innerHTML = '<option value="-1">No Insertion (Only Reorder/Delete)</option><option value="0">At the Beginning</option>';
    
    mainPageOrder.forEach((pageNum, index) => {
        const option = document.createElement('option');
        option.value = index + 1; // Position after this page
        option.textContent = `After Page ${pageNum}`;
        if (index === mainPageOrder.length - 1) {
            option.textContent += ' (At the End)';
        }
        positionSelect.appendChild(option);
    });
}


// Handle main PDF selection
async function handleMainPdfSelect() {
    const fileInput = document.getElementById('insertPdf-main-file');
    const fileNameSpan = document.getElementById('insertPdf-main-file-name');
    const pagesSpan = document.getElementById('insertPdf-main-pages');
    const previewsDiv = document.getElementById('insertPdf-previews');
    const pageList = document.getElementById('insertPdf-page-list');
    const positionSelect = document.getElementById('insertPdf-position');
    const progressDiv = document.getElementById('progress-insertPdfForm');
    const progressText = document.getElementById('progress-text-insertPdfForm');

    if (!fileInput.files[0]) return;

    // File size validation
    const maxSizeMB = 200;
    const fileSizeMB = fileInput.files[0].size / (1024 * 1024);
    if (fileSizeMB > maxSizeMB) {
        alert(`⚠️ File too large! Please upload a PDF smaller than ${maxSizeMB} MB.`);
        fileInput.value = "";
        return;
    }

    try {
        // const [pdfjs] = await pdfLibraryManager.loadLibraries(['pdfjs']);
        const [pdfjs, pdfLib] = await pdfLibraryManager.loadLibraries([
            'pdfjs', 'pdfLib'
        ]);

        
        // Show progress
        progressDiv.style.display = 'block';
        progressText.textContent = 'Loading main PDF...';

        const arrayBuffer = await fileInput.files[0].arrayBuffer();
        mainPDFDoc = await pdfjs.getDocument({ data: arrayBuffer }).promise;
        
        const numPages = mainPDFDoc.numPages;
        pagesSpan.textContent = `Total Pages: ${numPages}`;
        fileNameSpan.textContent = fileInput.files[0].name;

        // Initialize page order
        mainPageOrder = Array.from({length: numPages}, (_, i) => i + 1);
        selectedMainPages.clear();

        // Update position dropdown
        updatePositionDropdown();

        // Load page previews
        await loadMainPdfPreviews();

        previewsDiv.classList.remove('hidden');
        progressDiv.style.display = 'none';

    } catch (error) {
        console.error('Error loading main PDF:', error);
        progressDiv.style.display = 'none';
        alert('Error loading PDF: ' + error.message);
    }
}




async function loadMainPdfPreviews() {
    const pageList = document.getElementById('insertPdf-page-list');
    const positionSelect = document.getElementById('insertPdf-position');
    const selectedPosition = parseInt(positionSelect.value);

    // Show loading
    pageList.innerHTML = '<div class="col-span-3 text-center py-8"><div class="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto"></div><p class="text-gray-600 mt-2">Loading pages...</p></div>';

    let html = '';

    for (let i = 0; i < mainPageOrder.length; i++) {
        const originalPageNum = mainPageOrder[i];
        const page = await mainPDFDoc.getPage(originalPageNum);
        const viewport = page.getViewport({ scale: 0.3 });

        const canvas = document.createElement('canvas');
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        const context = canvas.getContext('2d', { willReadFrequently: true });

        // White background
        context.fillStyle = 'white';
        context.fillRect(0, 0, canvas.width, canvas.height);

        await page.render({ canvasContext: context, viewport }).promise;
        const canvasDataUrl = canvas.toDataURL();

        const showInsertionLine = selectedPosition === 0 ? i === 0 : i === selectedPosition - 1;
        const isSelected = selectedMainPages.has(originalPageNum);
        
        html += `
        <div class="page-insert-item border-2 ${isSelected ? 'border-red-500 bg-red-50' : showInsertionLine ? 'border-green-500 bg-green-50' : 'border-gray-200'} 
                    rounded-lg p-3 bg-white cursor-grab draggable-page" 
            data-original-page="${originalPageNum}" 
            data-current-index="${i}"
            draggable="true"
            onclick="togglePageSelection(${originalPageNum}, event)">
            <div class="flex flex-col items-center">
                <!-- Page Preview -->
                <div class="mb-2 relative">
                    <img src="${canvasDataUrl}" alt="Page ${originalPageNum}" 
                        class="border border-gray-300 rounded max-w-full h-auto">
                    <!-- ALWAYS show delete cross icon -->
                    <div class="absolute top-0 right-0 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs cursor-pointer hover:bg-red-600" 
                         onclick="event.stopPropagation(); removeSinglePage(${originalPageNum})">
                        <i class="fas fa-times"></i>
                    </div>
                </div>
                
                <!-- Page info and controls -->
                <div class="flex items-center justify-between w-full">
                    <span class="text-gray-700 font-medium">Page ${originalPageNum}</span>
                    <div class="flex space-x-1">
                 
                 
                    <!-- Move Up Button -->
<button type="button" onclick="event.stopPropagation(); movePageUp(${i})" 
        class="text-white rounded text-s ${i === 0 ? 'opacity-50 cursor-not-allowed' : ''}" 
        style="background-color: #3b82f6; border: none; margin-right: 14px; padding: 6px 10px;"
        ${i === 0 ? 'disabled' : ''}>
    <i class="fas fa-arrow-up"></i>
</button>

<!-- Move Down Button -->
<button type="button" onclick="event.stopPropagation(); movePageDown(${i})" 
        class="text-white rounded text-s ${i === mainPageOrder.length - 1 ? 'opacity-50 cursor-not-allowed' : ''}" 
        style="background-color: #3b82f6; border: none; padding: 6px 10px;"
        ${i === mainPageOrder.length - 1 ? 'disabled' : ''}>
    <i class="fas fa-arrow-down"></i>
</button>
                    </div>
                </div>
                
                <!-- Insertion line for visual indication -->
                ${showInsertionLine ? `
                    <div class="w-full mt-2 py-1 ${selectedPosition === 0 ? 'bg-blue-100 border-blue-300' : 'bg-green-100 border-green-300'} rounded text-center">
                        <span class="${selectedPosition === 0 ? 'text-blue-700' : 'text-green-700'} text-xs font-bold">
                            ${selectedPosition === 0 ? '▼ PDF will be inserted before this page ▼' : '▼ PDF will be inserted after this page ▼'}
                        </span>
                    </div>
                    ` : ''}
                
                <!-- Selection indicator -->
                ${isSelected ? `
                <div class="w-full mt-1 py-1 bg-red-100 border border-red-300 rounded text-center">
                    <span class="text-red-700 text-xs font-bold">SELECTED FOR DELETION</span>
                </div>
                ` : ''}
            </div>
        </div>
        `;
        canvas.remove();
    }

    pageList.innerHTML = html;
    
    // Initialize drag and drop
    initializeDragAndDrop();
}
// Initialize drag and drop functionality
function initializeDragAndDrop() {
    const draggables = document.querySelectorAll('.draggable-page');
    
    draggables.forEach(draggable => {
        draggable.addEventListener('dragstart', handleDragStart);
        draggable.addEventListener('dragover', handleDragOver);
        draggable.addEventListener('dragenter', handleDragEnter);
        draggable.addEventListener('dragleave', handleDragLeave);
        draggable.addEventListener('drop', handleDrop);
        draggable.addEventListener('dragend', handleDragEnd);
    });
}

// Drag and drop event handlers
function handleDragStart(e) {
    dragSrcEl = this;
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/html', this.innerHTML);
    this.classList.add('opacity-50', 'cursor-grabbing');
}

function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    return false;
}

function handleDragEnter(e) {
    this.classList.add('bg-blue-100', 'border-blue-400');
}

function handleDragLeave(e) {
    this.classList.remove('bg-blue-100', 'border-blue-400');
}

function handleDrop(e) {
    e.stopPropagation();
    e.preventDefault();
    
    if (dragSrcEl !== this) {
        const dragIndex = parseInt(dragSrcEl.dataset.currentIndex);
        const dropIndex = parseInt(this.dataset.currentIndex);
        
        // Reorder the pages array
        const [movedPage] = mainPageOrder.splice(dragIndex, 1);
        mainPageOrder.splice(dropIndex, 0, movedPage);
        
        // Use DOM manipulation instead of reloading previews
        const pageItems = document.querySelectorAll('.page-insert-item');
        const movedElement = pageItems[dragIndex];
        
        if (dropIndex > dragIndex) {
            this.parentNode.insertBefore(movedElement, pageItems[dropIndex].nextSibling);
        } else {
            this.parentNode.insertBefore(movedElement, pageItems[dropIndex]);
        }
        
        // Update indices and buttons
        updatePageIndicesAndButtons();
        updatePositionDropdown();
    }
    
    return false;
}
function handleDragEnd(e) {
    document.querySelectorAll('.draggable-page').forEach(el => {
        el.classList.remove('opacity-50', 'cursor-grabbing', 'bg-blue-100', 'border-blue-400');
    });
}

// Page movement functions
// Page movement functions with DOM manipulation
function movePageUp(index) {
    if (index > 0) {
        // Update the data array
        [mainPageOrder[index], mainPageOrder[index - 1]] = [mainPageOrder[index - 1], mainPageOrder[index]];
        
        // DOM manipulation - swap elements
        const pageItems = document.querySelectorAll('.page-insert-item');
        const currentItem = pageItems[index];
        const previousItem = pageItems[index - 1];
        
        // Swap the elements in DOM
        previousItem.parentNode.insertBefore(currentItem, previousItem);
        
        // Update indices and buttons
        updatePageIndicesAndButtons();
        updatePositionDropdown();
    }
}

function movePageDown(index) {
    if (index < mainPageOrder.length - 1) {
        // Update the data array
        [mainPageOrder[index], mainPageOrder[index + 1]] = [mainPageOrder[index + 1], mainPageOrder[index]];
        
        // DOM manipulation - move element down
        const pageItems = document.querySelectorAll('.page-insert-item');
        const currentItem = pageItems[index];
        const nextItem = pageItems[index + 1];
        
        // Move current item after next item
        if (nextItem.nextSibling) {
            nextItem.parentNode.insertBefore(currentItem, nextItem.nextSibling);
        } else {
            nextItem.parentNode.appendChild(currentItem);
        }
        
        // Update indices and buttons
        updatePageIndicesAndButtons();
        updatePositionDropdown();
    }
}



// Toggle page selection for deletion - without reloading

function togglePageSelection(pageNum, event) {
    if (selectedMainPages.has(pageNum)) {
        selectedMainPages.delete(pageNum);
    } else {
        selectedMainPages.add(pageNum);
    }
    
    // Update UI without reloading previews
    const pageItems = document.querySelectorAll('.page-insert-item');
    pageItems.forEach(item => {
        const itemPageNum = parseInt(item.dataset.originalPage);
        const isSelected = selectedMainPages.has(itemPageNum);
        
        // Update border and background
        item.classList.toggle('border-red-500', isSelected);
        item.classList.toggle('bg-red-50', isSelected);
        item.classList.toggle('border-gray-200', !isSelected);
        
        // Find or create selection indicator
        let selectionIndicator = item.querySelector('.selection-indicator');
        
        if (isSelected && !selectionIndicator) {
            // Add selection indicator
            selectionIndicator = document.createElement('div');
            selectionIndicator.className = 'w-full mt-1 py-1 bg-red-100 border border-red-300 rounded text-center selection-indicator';
            // selectionIndicator.innerHTML = '<span class="text-red-700 text-xs font-bold">SELECTED FOR DELETION</span>';
            selectionIndicator.innerHTML = `
                <span class="text-red-700 text-xs font-bold">SELECTED FOR DELETION</span>
                <span class="text-green-700 text-xs block mt-1">CLICK AGAIN TO DE SELECT</span>
            `;
                        // Insert before the insertion line or at the end
            const insertionLine = item.querySelector('.insertion-line');
            if (insertionLine) {
                insertionLine.parentNode.insertBefore(selectionIndicator, insertionLine);
            } else {
                item.querySelector('.flex.flex-col.items-center').appendChild(selectionIndicator);
            }
        } else if (!isSelected && selectionIndicator) {
            // Remove selection indicator
            selectionIndicator.remove();
        }
    });
}


// function togglePageSelection(pageNum, event) {
//     if (selectedMainPages.has(pageNum)) {
//         selectedMainPages.delete(pageNum);
//     } else {
//         selectedMainPages.add(pageNum);
//     }
    
//     // Update UI without reloading previews
//     const pageItems = document.querySelectorAll('.page-insert-item');
//     pageItems.forEach(item => {
//         const itemPageNum = parseInt(item.dataset.originalPage);
//         if (selectedMainPages.has(itemPageNum)) {
//             item.classList.add('border-red-500', 'bg-red-50');
//             item.classList.remove('border-gray-200');
//         } else {
//             item.classList.remove('border-red-500', 'bg-red-50');
//             item.classList.add('border-gray-200');
//         }
//     });
// }

// Select/Deselect all pages - without reloading
function toggleSelectAllMainPages() {
    const button = document.getElementById('select-all-main-pages-btn');
    const allSelected = selectedMainPages.size === mainPageOrder.length;
    
    if (allSelected) {
        // Deselect all
        selectedMainPages.clear();
        button.innerHTML = '<i class="fas fa-check-square mr-2"></i> Select All';
        button.classList.remove('bg-green-600', 'hover:bg-green-700');
        button.classList.add('bg-gray-600', 'hover:bg-gray-700');
    } else {
        // Select all
        mainPageOrder.forEach(pageNum => selectedMainPages.add(pageNum));
        button.innerHTML = '<i class="fas fa-times-circle mr-2"></i> Deselect All';
        button.classList.remove('bg-gray-600', 'hover:bg-gray-700');
        button.classList.add('bg-green-600', 'hover:bg-green-700');
    }
    
    // Update UI without reloading previews
    const pageItems = document.querySelectorAll('.page-insert-item');
    pageItems.forEach(item => {
        const itemPageNum = parseInt(item.dataset.originalPage);
        if (selectedMainPages.has(itemPageNum)) {
            item.classList.add('border-red-500', 'bg-red-50');
            item.classList.remove('border-gray-200');
        } else {
            item.classList.remove('border-red-500', 'bg-red-50');
            item.classList.add('border-gray-200');
        }
    });
}



// Delete selected pages
function deleteSelectedMainPages() {
    if (selectedMainPages.size === 0) {
        alert('Please select pages to delete by clicking on them.');
        return;
    }
    
    const deleteCount = selectedMainPages.size; // Store the count BEFORE clearing
    
    if (confirm(`Are you sure you want to delete ${deleteCount} page(s)?`)) {
        // Remove selected pages from the order array
        mainPageOrder = mainPageOrder.filter(pageNum => !selectedMainPages.has(pageNum));
        
        // Update UI
        loadMainPdfPreviews();
        updatePositionDropdown();
        
        const resultDiv = document.getElementById('result-insertPdfForm');
        resultDiv.innerHTML = `<div class="text-green-600">✅ Deleted ${deleteCount} page(s). Reordered pages preserved.</div>`;
        
        // Clear selection AFTER we've used the count
        selectedMainPages.clear();
        
        // Reset select all button
        const button = document.getElementById('select-all-main-pages-btn');
        button.innerHTML = '<i class="fas fa-check-square mr-2"></i> Select All';
        button.classList.remove('bg-green-600', 'hover:bg-green-700');
        button.classList.add('bg-gray-600', 'hover:bg-gray-700');
    }
}
// Reset to original order
function resetMainPagesOrder() {
    if (mainPDFDoc) {
        mainPageOrder = Array.from({length: mainPDFDoc.numPages}, (_, i) => i + 1);
        selectedMainPages.clear();
        loadMainPdfPreviews();
        updatePositionDropdown();
        
        const resultDiv = document.getElementById('result-insertPdfForm');
        resultDiv.innerHTML = `<div class="text-blue-600">✅ Page order reset to original.</div>`;
        
        // Reset select all button
        const button = document.getElementById('select-all-main-pages-btn');
        button.innerHTML = '<i class="fas fa-check-square mr-2"></i> Select All';
        button.classList.remove('bg-green-600', 'hover:bg-green-700');
        button.classList.add('bg-gray-600', 'hover:bg-gray-700');
    }
}

// Remove single page when cross icon is clicked
function removeSinglePage(pageNum) {
    if (confirm(`Are you sure you want to delete Page ${pageNum}?`)) {
        // Remove the page from the order array
        mainPageOrder = mainPageOrder.filter(p => p !== pageNum);
        
        // Remove from selection if it was selected
        selectedMainPages.delete(pageNum);
        
        // Update UI
        loadMainPdfPreviews();
        updatePositionDropdown();
        
        const resultDiv = document.getElementById('result-insertPdfForm');
        resultDiv.innerHTML = `<div class="text-green-600">✅ Deleted Page ${pageNum}. Reordered pages preserved.</div>`;
        
        // Update select all button state
        updateSelectAllButtonState();
    }
}

// Helper function to update select all button state
function updateSelectAllButtonState() {
    const button = document.getElementById('select-all-main-pages-btn');
    const allSelected = selectedMainPages.size === mainPageOrder.length;
    
    if (allSelected && mainPageOrder.length > 0) {
        button.innerHTML = '<i class="fas fa-times-circle mr-2"></i> Deselect All';
        button.classList.remove('bg-gray-600', 'hover:bg-gray-700');
        button.classList.add('bg-green-600', 'hover:bg-green-700');
    } else {
        button.innerHTML = '<i class="fas fa-check-square mr-2"></i> Select All';
        button.classList.remove('bg-green-600', 'hover:bg-green-700');
        button.classList.add('bg-gray-600', 'hover:bg-gray-700');
    }
}
// Enhanced main insertion function with reordering and deletion
async function insertPDFClientSide() {
    const mainFileInput = document.getElementById('insertPdf-main-file');
    const insertFileInput = document.getElementById('insertPdf-insert-file');
    const positionSelect = document.getElementById('insertPdf-position');
    const resultDiv = document.getElementById('result-insertPdfForm');
    const progressDiv = document.getElementById('progress-insertPdfForm');
    const progressText = document.getElementById('progress-text-insertPdfForm');
    const genpdfbutton= document.getElementById('genpdf');

    // Validation
    if (!mainFileInput.files[0]) {
        alert('Please select main PDF file.');
        return;
    }

    if (!mainPDFDoc) {
        alert('Please wait for main PDF to load completely.');
        return;
    }

    const insertPosition = parseInt(positionSelect.value);
    const wantsInsertion = insertPosition !== -1 && insertFileInput.files[0];

    // Validate insert PDF only if insertion is requested
    if (wantsInsertion && !insertPDFDoc) {
        alert('Please wait for PDF to insert to load completely.');
        return;
    }

    const insertPages = wantsInsertion ? insertPDFDoc.numPages : 0;

    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Starting PDF processing...';


    try {
        const { PDFDocument } = pdfLibraryManager.libraries.pdfLib.lib;

        progressText.textContent = 'Loading main PDF...';
        genpdfbutton.disabled = true;
        genpdfbutton.textContent = 'Processing...';
       

        // Get original bytes for main PDF
        const mainArrayBuffer = await mainPDFDoc.getData();
        const mainPdfDoc = await PDFDocument.load(mainArrayBuffer);

        // Load insert PDF only if insertion is requested
        let insertPdfDoc = null;
        if (wantsInsertion) {
            progressText.textContent = 'Loading insert PDF...';
            const insertArrayBuffer = await insertPDFDoc.getData();
            insertPdfDoc = await PDFDocument.load(insertArrayBuffer);
        }

        progressText.textContent = 'Creating new PDF with reordering and insertion...';

        // Create new PDF document
        const newPdfDoc = await PDFDocument.create();

        // Copy pages from main PDF according to current order and deletion
        const finalMainPages = mainPageOrder;

        console.log('Processing:', {
            wantsInsertion,
            insertPosition,
            finalMainPages,
            mainPages: mainPdfDoc.getPageCount()
        });

        // FIX: When no insertion, copy ALL pages in the reordered sequence
        if (!wantsInsertion) {
            // Just copy all reordered pages (no insertion)
            for (let i = 0; i < finalMainPages.length; i++) {
                progressText.textContent = `Copying pages... (${i + 1}/${finalMainPages.length})`;
                const originalPageIndex = finalMainPages[i] - 1;
                
                if (originalPageIndex >= 0 && originalPageIndex < mainPdfDoc.getPageCount()) {
                    const [copiedPage] = await newPdfDoc.copyPages(mainPdfDoc, [originalPageIndex]);
                    newPdfDoc.addPage(copiedPage);
                }
            }
        } else {
            // Original logic with insertion
            // Copy pages before insertion point
            for (let i = 0; i < insertPosition; i++) {
                progressText.textContent = `Copying main PDF pages... (${i + 1}/${finalMainPages.length})`;
                const originalPageIndex = finalMainPages[i] - 1;
                
                if (originalPageIndex >= 0 && originalPageIndex < mainPdfDoc.getPageCount()) {
                    const [copiedPage] = await newPdfDoc.copyPages(mainPdfDoc, [originalPageIndex]);
                    newPdfDoc.addPage(copiedPage);
                }
            }

            // Copy pages from insert PDF
            if (insertPdfDoc) {
                for (let i = 0; i < insertPages; i++) {
                    progressText.textContent = `Copying inserted PDF pages... (${i + 1}/${insertPages})`;
                    const [copiedPage] = await newPdfDoc.copyPages(insertPdfDoc, [i]);
                    newPdfDoc.addPage(copiedPage);
                }
            }

            // Copy remaining pages from main PDF after insertion point
            for (let i = insertPosition; i < finalMainPages.length; i++) {
                progressText.textContent = `Copying remaining main PDF pages... (${i + 1}/${finalMainPages.length})`;
                const originalPageIndex = finalMainPages[i] - 1;
                
                if (originalPageIndex >= 0 && originalPageIndex < mainPdfDoc.getPageCount()) {
                    const [copiedPage] = await newPdfDoc.copyPages(mainPdfDoc, [originalPageIndex]);
                    newPdfDoc.addPage(copiedPage);
                }
            }
        }

        progressText.textContent = 'Saving final PDF...';

        // Save the new PDF
        const newPdfBytes = await newPdfDoc.save();
        const newPdfBlob = new Blob([newPdfBytes], { type: 'application/pdf' });

        // Generate filename
        const mainFileName = mainFileInput.files[0].name.replace('.pdf', '');
        let filename;
        if (wantsInsertion) {
            const insertFileName = insertFileInput.files[0].name.replace('.pdf', '');
            filename = `final_${mainFileName}_with_${insertFileName}.pdf`;
        } else {
            filename = `reordered_${mainFileName}.pdf`;
        }

        // Download
        const url = URL.createObjectURL(newPdfBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        // Update result message
        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>Final PDF Generated Successfully!</strong><br>
                📁 File: ${filename}<br>
                📄 Original Main PDF: ${mainPDFDoc.numPages} pages<br>
                📄 Final Main Pages: ${finalMainPages.length} pages (after reordering/deletion)<br>
                ${wantsInsertion ? `📄 Inserted PDF: ${insertPages} pages<br>` : ''}
                ${wantsInsertion ? `📍 Inserted after position: ${insertPosition}<br>` : ''}
                📊 Total pages in result: ${finalMainPages.length + (wantsInsertion ? insertPages : 0)}<br>
                🔄 Pages reordered: ${mainPDFDoc.numPages !== finalMainPages.length ? 'Yes' : 'No'}<br>
                🗑️ Pages deleted: ${mainPDFDoc.numPages - finalMainPages.length}
            </div>
        `;

        progressDiv.style.display = 'none';
        genpdfbutton.disabled = false;
        // genpdfbutton.textContent = 'Generate PDF';
        genpdfbutton.innerHTML = `<i class="fas fa-file-import mr-2"></i> Generate PDF`;

    } catch (error) {
        console.error('PDF processing failed:', error);
        progressDiv.style.display = 'none';
        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Processing failed: ${error.message}<br>
                <small>Please try with a different PDF file.</small>
            </div>
        `;
    }
}
document.getElementById('insertPdf-insert-file').addEventListener('change', async function(e) {
    const fileInput = e.target;
    const fileNameSpan = document.getElementById('insertPdf-insert-file-name');
    const pagesSpan = document.getElementById('insertPdf-insert-pages');

    if (!fileInput.files[0]) {
        // Clear the insert PDF doc if no file selected
        insertPDFDoc = null;
        pagesSpan.textContent = 'Total Pages: Not loaded';
        fileNameSpan.textContent = 'No file selected';
        return;
    }

    // File size validation
    const maxSizeMB = 100;
    const fileSizeMB = fileInput.files[0].size / (1024 * 1024);
    if (fileSizeMB > maxSizeMB) {
        alert(`⚠️ File too large! Please upload a PDF smaller than ${maxSizeMB} MB.`);
        fileInput.value = "";
        insertPDFDoc = null;
        pagesSpan.textContent = 'Total Pages: Not loaded';
        fileNameSpan.textContent = 'No file selected';
        return;
    }

    try {
        const [pdfjs] = await pdfLibraryManager.loadLibraries(['pdfjs']);
        const arrayBuffer = await fileInput.files[0].arrayBuffer();
        insertPDFDoc = await pdfjs.getDocument({ data: arrayBuffer }).promise;
        
        const numPages = insertPDFDoc.numPages;
        pagesSpan.textContent = `Total Pages: ${numPages}`;
        fileNameSpan.textContent = fileInput.files[0].name;

    } catch (error) {
        console.error('Error loading insert PDF:', error);
        alert('Error loading PDF to insert: ' + error.message);
        insertPDFDoc = null;
        pagesSpan.textContent = 'Total Pages: Not loaded';
    }
});

// Update insertion point when position changes
document.getElementById('insertPdf-position').addEventListener('change', function() {
    if (mainPDFDoc) {
        loadMainPdfPreviews();
    }
});

// Update file label function
updateFileLabel('insertPdf-main-file', 'insertPdf-main-file-name');
updateFileLabel('insertPdf-insert-file', 'insertPdf-insert-file-name');



